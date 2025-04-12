from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffgar.utils.subject_processing import *
from pytorch_lightning import LightningModule

from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from .unet import UNet
from pytorch_lightning.cli import OptimizerCallable
from torch import optim
from diffgar.models.utils.schedulers import *
from .text_encoders import T5TextEncoder
import numpy as np
import yaml
import scipy



from diffgar.models.clap.src.laion_clap import CLAP_Module
from diffgar.models.muleT5.muleT5 import MuleT5EncoderPair
from diffgar.models.muscall.muscall.models.muscall import MusCALL
from diffgar.models.muleT5.muleT5 import MuleCLAPEncoderPair
from diffgar.models.utils.losses import NTXent
from torchaudio.functional import frechet_distance




def configure_optimizers_(self_):
    if self_.optimizer is None:
        optimizer = optim.Adam(
            self_.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = self_.optimizer(self_.parameters())
        
    if self_.scheduler is not None:
        # copy of the scheduler applied to the optimizer
        scheduler_class = eval(self_.scheduler['class_name'])
        scheduler_kwargs = self_.scheduler.get('init_args', {})
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        self_.scheduler = scheduler
        return [optimizer], [scheduler]
    
    return optimizer

# def KLD(p, q):
#     #KL divergence between two sampled distributions
#     ## can be of different shapes (if so, restrict to the minimum shape)
    
#     N = min(p.shape[0], q.shape[0])
#     if p.shape[0] > N:
#         p = p[:N]
#     if q.shape[0] > N:
#         q = q[:N]
        
#     return F.kl_div(F.log_softmax(p, dim=1), F.softmax(q, dim=1), reduction='batchmean')

def FD(p, q):
    
    #shapes (batch_size, num_channels)
    mu_p = p.mean(0)
    mu_q = q.mean(0)
    
    sigma_p = torch.cov(p.T)
    sigma_q = torch.cov(q.T)
    
    return frechet_distance(mu_p, sigma_p, mu_q, sigma_q)
    
def on_validation_epoch_end_(self_, thresholds = [1,5,10]):
    
    
    #only every 50 epochs
    if self_.current_epoch % self_.retrieval_every_n_epochs != 0:
        return
    
    # KL divergence and frechet distance between distributions of train text and audio embeddings and generated samples
    train_text, train_audio = torch.cat(self_.train_gt_text, dim=0), torch.cat(self_.train_gt_audio, dim=0)
    
    
        
    for i in range(len(self_.val_audio_preds)):
        if len(self_.val_audio_preds[i]) == 0:
            return
        
        dataloader_idx = i
        
        
        audio_preds = torch.cat(self_.val_audio_preds[i], dim=0)
        gt_audio = torch.cat(self_.val_gt_audio[i], dim=0)
        gt_text = torch.cat(self_.val_gt_text[i], dim=0)
        
        #get the name from the dataloader index
        dataloader_name = self_.trainer.datamodule.dataloader_names[dataloader_idx]
        
        # kld_audio = KLD(gt_audio, train_audio)
        # kld_text = KLD(gt_text, train_text)
        
        fd_audio = FD(gt_audio, train_audio)
        fd_text = FD(gt_text, train_text)
        
        # for metric, value in zip(['KL_Divergence/Audio', 'KL_Divergence/Text', 'Frechet_Distance/Audio', 'Frechet_Distance/Text'], [kld_audio, kld_text, fd_audio, fd_text]):
        #     self_.log(f'{dataloader_name}_{metric}', value, on_step=False, on_epoch=True, prog_bar=False)
    
        for metric, value in zip(['Frechet_Distance/Audio', 'Frechet_Distance/Text'], [fd_audio, fd_text]):
            self_.log(f'{dataloader_name}_{metric}', value, on_step=False, on_epoch=True, prog_bar=False)
    
        #get the name from the dataloader index
        dataloader_name = self_.trainer.datamodule.dataloader_names[dataloader_idx]
        
        
        audio_preds = audio_preds / audio_preds.norm(dim=1, keepdim=True)
        gt_audio = gt_audio / gt_audio.norm(dim=1, keepdim=True)
        gt_text = gt_text / gt_text.norm(dim=1, keepdim=True)
        
        pred_text_2_audio = audio_preds @ gt_audio.t() if audio_preds.shape[-1] == gt_audio.shape[-1] else torch.zeros((audio_preds.shape[0], gt_audio.shape[0]), device = audio_preds.device)
        gt_text_2_audio = gt_text @ gt_audio.t() if gt_text.shape[-1] == gt_audio.shape[-1] else torch.zeros((gt_text.shape[0], gt_audio.shape[0]), device = gt_text.device)
        pred_text_2_text = audio_preds @ gt_text.t() if audio_preds.shape[-1] == gt_text.shape[-1] else torch.zeros((audio_preds.shape[0], gt_text.shape[0]), device = audio_preds.device)
        gt_audio_2_text = gt_audio @ gt_text.t() if gt_audio.shape[-1] == gt_text.shape[-1] else torch.zeros((gt_audio.shape[0], gt_text.shape[0]), device = gt_audio.device)
        
        
        
        print(f'========================== {dataloader_name} ==========================')
        print(f'Audio -> Text: {gt_audio_2_text.diag().mean()}')
        print(f'Text -> Audio: {gt_text_2_audio.diag().mean()}')
        print(f'Pred Text -> Text: {pred_text_2_text.diag().mean()}')
        print(f'Pred Text -> Audio: {pred_text_2_audio.diag().mean()}')
        print(f'============================================================')
        
        
        t2a_recall, t2a_precision, t2a_ranks, t2a_normalized_ranks = compute_recall(gt_text_2_audio)
        a2t_recall, a2t_precision, a2t_ranks, a2t_normalized_ranks = compute_recall(gt_audio_2_text)
        t2t_recall, t2t_precision, t2t_ranks, t2t_normalized_ranks = compute_recall(pred_text_2_text)
        t2a_preds_recall, t2a_preds_precision, t2a_preds_ranks, t2a_preds_normalized_ranks = compute_recall(pred_text_2_audio)
        
        #get dataloader names from datamodule
        dataloader_names = self_.trainer.datamodule.dataloader_names
        
        for threshold in thresholds:
            self_.log(f'{dataloader_names[dataloader_idx]}_Retrieval/Val/T->A_recall@{threshold}', t2a_recall[threshold], on_step=False, on_epoch=True, prog_bar=False)
            self_.log(f'{dataloader_names[dataloader_idx]}_Retrieval/Val/A->T_recall@{threshold}', a2t_recall[threshold], on_step=False, on_epoch=True, prog_bar=False)
            self_.log(f'{dataloader_names[dataloader_idx]}_Retrieval/Val/predT->T_recall@{threshold}', t2t_recall[threshold], on_step=False, on_epoch=True, prog_bar=False)
            self_.log(f'{dataloader_names[dataloader_idx]}_Retrieval/Val/predT->A_recall@{threshold}', t2a_preds_recall[threshold], on_step=False, on_epoch=True, prog_bar=False)
            
            ## median ranks
            self_.log(f'{dataloader_names[dataloader_idx]}_Retrieval/Val/T->A_median_rank', t2a_normalized_ranks.median(), on_step=False, on_epoch=True, prog_bar=False)
            self_.log(f'{dataloader_names[dataloader_idx]}_Retrieval/Val/A->T_median_rank', a2t_normalized_ranks.median(), on_step=False, on_epoch=True, prog_bar=False)
            self_.log(f'{dataloader_names[dataloader_idx]}_Retrieval/Val/predT->T_median_rank', t2t_normalized_ranks.median(), on_step=False, on_epoch=True, prog_bar=False)
            self_.log(f'{dataloader_names[dataloader_idx]}_Retrieval/Val/predT->A_median_rank', t2a_preds_normalized_ranks.median(), on_step=False, on_epoch=True, prog_bar=False)
        
        ## reset the lists
        self_.val_audio_preds[i] = []
        self_.val_gt_audio[i] = []
        self_.val_gt_text[i] = []
        
    
    self_.train_audio_preds = []
    self_.train_gt_audio = []
    self_.train_gt_text = []
    self_.train_prompts = []
    
def on_train_epoch_end_(self_, thresholds = [1,5,10]):
    
    if len(self_.train_audio_preds) == 0:
        return

    if self_.current_epoch % self_.retrieval_every_n_epochs != 0:
        return

    ## retrieval @1,5,10 on the training set using the generated samples
    gt_audio = torch.cat(self_.train_gt_audio, dim=0)
    gt_text = torch.cat(self_.train_gt_text, dim=0)
    audio_preds = torch.cat(self_.train_audio_preds, dim=0)
    
    audio_preds = audio_preds / audio_preds.norm(dim=1, keepdim=True)
    gt_audio = gt_audio / gt_audio.norm(dim=1, keepdim=True)
    gt_text = gt_text / gt_text.norm(dim=1, keepdim=True)
    
    pred_text_2_audio = audio_preds @ gt_audio.t() if audio_preds.shape[-1] == gt_audio.shape[-1] else torch.zeros((audio_preds.shape[0], gt_audio.shape[0]), device = audio_preds.device)
    gt_text_2_audio = gt_text @ gt_audio.t() if gt_text.shape[-1] == gt_audio.shape[-1] else torch.zeros((gt_text.shape[0], gt_audio.shape[0]), device = gt_text.device)
    pred_text_2_text = audio_preds @ gt_text.t() if audio_preds.shape[-1] == gt_text.shape[-1] else torch.zeros((audio_preds.shape[0], gt_text.shape[0]), device = audio_preds.device)
    gt_audio_2_text = gt_audio @ gt_text.t() if gt_audio.shape[-1] == gt_text.shape[-1] else torch.zeros((gt_audio.shape[0], gt_text.shape[0]), device = gt_audio.device)
    
    print(f'========================== Training Set ==========================')
    print(f'Audio -> Text: {gt_audio_2_text.diag().mean()}')
    print(f'Text -> Audio: {gt_text_2_audio.diag().mean()}')
    print(f'Pred Text -> Text: {pred_text_2_text.diag().mean()}')
    print(f'Pred Text -> Audio: {pred_text_2_audio.diag().mean()}')
    print(f'============================================================')
    
    t2a_recall, t2a_precision, t2a_ranks, t2a_normalized_ranks = compute_recall(gt_text_2_audio)
    a2t_recall, a2t_precision, a2t_ranks, a2t_normalized_ranks = compute_recall(gt_audio_2_text)
    t2t_recall, t2t_precision, t2t_ranks, t2t_normalized_ranks = compute_recall(pred_text_2_text)
    t2a_preds_recall, t2a_preds_precision, t2a_preds_ranks, t2a_preds_normalized_ranks = compute_recall(pred_text_2_audio)
    
    for threshold in thresholds:
        self_.log(f'Train_Retrieval/T->A_recall@{threshold}', t2a_recall[threshold], on_step=False, on_epoch=True, prog_bar=False)
        self_.log(f'Train_Retrieval/A->T_recall@{threshold}', a2t_recall[threshold], on_step=False, on_epoch=True, prog_bar=False)
        self_.log(f'Train_Retrieval/predT->T_recall@{threshold}', t2t_recall[threshold], on_step=False, on_epoch=True, prog_bar=False)
        self_.log(f'Train_Retrieval/predT->A_recall@{threshold}', t2a_preds_recall[threshold], on_step=False, on_epoch=True, prog_bar=False)
        
        ## median ranks
        self_.log(f'Train_Retrieval/T->A_median_rank', t2a_normalized_ranks.median(), on_step=False, on_epoch=True, prog_bar=False)
        self_.log(f'Train_Retrieval/A->T_median_rank', a2t_normalized_ranks.median(), on_step=False, on_epoch=True, prog_bar=False)
        self_.log(f'Train_Retrieval/predT->T_median_rank', t2t_normalized_ranks.median(), on_step=False, on_epoch=True, prog_bar=False)
        self_.log(f'Train_Retrieval/predT->A_median_rank', t2a_preds_normalized_ranks.median(), on_step=False, on_epoch=True, prog_bar=False)
    
    
def get_encoder_pair(encoder_pair, encoder_pair_kwargs=None):
    
    if encoder_pair == "clap":
        return CLAP_Module(**encoder_pair_kwargs)
    elif encoder_pair == "muleT5":
        return MuleT5EncoderPair(**encoder_pair_kwargs)
    elif encoder_pair == "muleCLAP":
        return MuleCLAPEncoderPair(**encoder_pair_kwargs)
    elif encoder_pair == "MusCALL":
        return MusCALL.from_pretrained(**encoder_pair_kwargs)
    elif encoder_pair == "music2latent":
        return MuleT5EncoderPair(**encoder_pair_kwargs)
    
def get_text_encoder(text_encoder, text_encoder_kwargs=None):
    if text_encoder == 'T5':
        return T5TextEncoder(**text_encoder_kwargs)
    
    
def compute_recall(similarities: torch.Tensor):
        num_src_embeddings, num_tgt_embeddings = similarities.size()
        device = similarities.device
        

        true_indices = torch.arange(num_src_embeddings, device=device).unsqueeze(1)
        sorted_indices = similarities.argsort(descending=True)

        if num_src_embeddings < num_tgt_embeddings:
            tgt_per_src, r = divmod(num_tgt_embeddings, num_src_embeddings)
            assert r == 0
            sorted_indices = sorted_indices.div_(tgt_per_src, rounding_mode="floor")

        else:
            src_per_tgt, r = divmod(num_src_embeddings, num_tgt_embeddings)
            assert r == 0
            true_indices.div_(src_per_tgt, rounding_mode="floor")

        ranks = (sorted_indices == true_indices).long().argmax(dim=1)  # argmax?

        recalls = torch.zeros(num_tgt_embeddings + 1, dtype=torch.long, device=device)
        precisions = torch.zeros(num_tgt_embeddings + 1, dtype=torch.long, device=device)
        values, counts = torch.unique(ranks, return_counts=True)
        recalls[values + 1] = counts
        precisions[values + 1] = counts.cumsum(dim=0)
        recalls = recalls.cumsum(dim=0).float().div_(num_src_embeddings)
        precisions = precisions.cumsum(dim=0).float().div_(num_src_embeddings)
        
        normalized_ranks = ranks.float().div_(ranks.max())
        
        return recalls, precisions, ranks, normalized_ranks



class BaseModule(nn.Module):
    
    def __init__(self):
        super().__init__()
    
         
    @classmethod
    def from_config(cls, config, device=None):
        config.update(device=device)
        return cls(**config)
    
    @classmethod
    def from_yaml(cls, yaml_path, device=None):
        
        if 's3://' in yaml_path:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
        else:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
        
        config = config.get('model', config)
        config = config.get('init_args', config)
        
        return cls.from_config(config, device=device)
    
    @classmethod
    def from_pretrained(cls, yaml_or_config, ckpt_path, device=None):
        if isinstance(yaml_or_config, str):
            model = cls.from_yaml(yaml_or_config, device=device)
        else:
            model = cls.from_config(yaml_or_config, device=device)
            
        if 's3://' in ckpt_path:
            from s3torchconnector import S3Checkpoint
            checkpoint= S3Checkpoint(region='us-east-1')
            with checkpoint.reader(ckpt_path) as f:
                ckpt = torch.load(f, map_location=device)
                model.load_state_dict(ckpt['state_dict'], strict=True)
                print(f"Model loaded from {ckpt_path}")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            print(f"Model loaded from {ckpt_path}")
        
        return model
    
     
    def freeze_text_encoder(self):
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
    def unfreeze_text_encoder(self):
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = True
            
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        
        for param in self.parameters():
            param.requires_grad = True
            
    
    def freeze_encoder_pair(self):
        for param in self.encoder_pair.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder_pair(self):
        for param in self.encoder_pair.parameters():
            param.requires_grad = True
            
            
    def encode_text(self, prompt, subject_masks = None):
        
        
        # if self.subject_flagging:
        #     offsets = self.text_encoder.get_text_embedding(prompt,return_dict = True, return_tokenizer_only = True)['offset_mapping'] if self.text_encoder is not None else self.encoder_pair.get_text_embedding(prompt, return_dict = True, return_tokenizer_only = True)['offset_mapping']
        #     prompt, subject_masks = create_binary_mask_from_list(prompt, offsets)

        if self.freeze_encoders:
            with torch.no_grad():
                encoded_text_dict = self.text_encoder.get_text_embedding(prompt, use_tensor = True, return_dict = True) if self.text_encoder is not None else self.encoder_pair.get_text_embedding(prompt, use_tensor = True, return_dict = True)
        else:
            encoded_text_dict = self.text_encoder.get_text_embedding(prompt, use_tensor = True, return_dict = True) if self.text_encoder is not None else self.encoder_pair.get_text_embedding(prompt, use_tensor = True, return_dict = True)
            
        if subject_masks is not None:
            encoded_text_dict['subject_masks'] = subject_masks

        return encoded_text_dict
            
class DiffGarLDM(BaseModule):
    def __init__(
        self,
        encoder_pair='clap', ## this will be used for clap score, but the text encoder can be different
        encoder_pair_kwargs=None,
        encoder_pair_ckpt=None,
        text_encoder = None,
        text_encoder_kwargs = None,
        text_encoder_ckpt = None,
        scheduler_name='stabilityai/stable-diffusion-2-1',
        scheduler_pred_type='epsilon',
        unet_model_config=None,
        unet_ckpt=None,
        freeze_encoder_pair=True,
        freeze_unet=False,
        device=None,
        contrastive_loss = False,
        contrastive_loss_kwargs = {
            'temperature': 0.1,
            'weight': 0.5,
        },
        **kwargs
    ):
        super().__init__()

        ## DIFFUSION SCHEDULERS

        self.scheduler_name = scheduler_name
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler", prediction_type=scheduler_pred_type)
        self.inference_scheduler = DDIMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler", prediction_type=scheduler_pred_type)
        self.noise_scheduler.prediction_type = scheduler_pred_type
        self.inference_scheduler.prediction_type = scheduler_pred_type
        self.noise_scheduler.register_to_config(prediction_type=scheduler_pred_type)
        self.inference_scheduler.register_to_config(prediction_type=scheduler_pred_type)
        
        
        ## UNET MODEL
        
        self.unet_model_config = unet_model_config
        self.max_length = self.unet_model_config['embedding_max_length'] if self.unet_model_config is not None else 0
        self.unet = UNet.from_config(self.unet_model_config) if self.unet_model_config is not None else None
        print(f"Diffusion model initialized with scheduler {self.noise_scheduler}")
        
        if unet_ckpt:
            ckpt = torch.load(unet_ckpt)
            self.unet.load_state_dict(ckpt['state_dict'])
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False


        ## ENCODERS

        self.encoder_pair = get_encoder_pair(encoder_pair, encoder_pair_kwargs)
        if encoder_pair_ckpt: self.encoder_pair.load_ckpt(encoder_pair_ckpt, verbose=False)
        
        self.text_encoder = None
        if text_encoder is not None:
            self.text_encoder = get_text_encoder(text_encoder, text_encoder_kwargs)
            self.text_encoder.max_length = self.max_length
            if text_encoder_ckpt:
                self.text_encoder.load_ckpt(text_encoder_ckpt)
        
        if freeze_encoder_pair:
            self.freeze_encoder_pair()
            self.freeze_text_encoder()
            
        self.freeze_encoders = freeze_encoder_pair
        with torch.no_grad():
            text_dim = self.text_encoder.get_text_embedding("test")['last_hidden_state'].shape[-1] if self.text_encoder is not None else self.encoder_pair.get_text_embedding("test")['last_hidden_state'].shape[-1]
        
        
        print(f"Text dimension: {text_dim}")
        
        ## OTHERS
    
        self.first_run = False
        self.contrastive_loss = None
        if contrastive_loss:
            self.contrastive_loss = NTXent(**contrastive_loss_kwargs)
            
        if device is not None:
            self.to(device)            
           

    def forward(self, latents, prompt, validation_mode=False):
        
        latents = latents.float()

        bsz = latents.shape[0]
        
        device = next(self.parameters()).device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)
        text_dict = self.encode_text(prompt)
        
        encoder_hidden_states = text_dict['last_hidden_state']

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()
                
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = latents
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        bsz, length, device = *encoder_hidden_states.shape[0:2], encoder_hidden_states.device
        
        assert latents.shape == noisy_latents.shape, "Latents and noisy latents shape mismatch"
        assert latents.shape == target.shape, "Latents and target shape mismatch"
        
        if self.unet_model_config['classifier_free_guidance_strength'] is not None:
            guidance_scale = self.unet_model_config['classifier_free_guidance_strength']
        else:
            guidance_scale = None
        
        model_pred = self.unet(
            noisy_latents, time = timesteps, embedding = encoder_hidden_states, embedding_mask_proba = guidance_scale, embedding_scale = 1.0
        )

        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        if self.contrastive_loss:
            contrastive_loss = self.contrastive_loss(model_pred, target, step = self.global_step)
            loss = mse_loss + contrastive_loss * self.contrastive_loss.weight
        else:
            loss = mse_loss
            
            
        loss_dict = {
            'mse_loss': mse_loss,
        }
        
        if self.contrastive_loss:
            loss_dict['contrastive_loss'] = contrastive_loss

        assert model_pred.shape == target.shape, "Model prediction and target shape mismatch"
        return loss, loss_dict

    @torch.no_grad()
    def inference(self, prompt, inference_scheduler = None, num_steps=20, guidance_scale=None, num_samples_per_prompt=1, 
                  disable_progress=True, slider = None, slider_scale = 0, negative_prompt = None, return_all_latents = False):
        
        
        device = next(self.parameters()).device
        batch_size = len(prompt) * num_samples_per_prompt
        
        inference_scheduler = self.inference_scheduler if inference_scheduler is None else inference_scheduler 
        
        encoded_text = self.encode_text(prompt)
        prompt_embeds = encoded_text['last_hidden_state']
        boolean_prompt_mask = encoded_text['attention_mask']
        subject_mask = encoded_text.get('subject_masks', None)
        
        if slider and subject_mask:
            prompt_embeds = slider.apply(prompt_embeds,subject_mask, scale = slider_scale)
        
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
        boolean_prompt_mask = (boolean_prompt_mask == 1).to(device)
        
        
        if negative_prompt:
            encoded_negative_text = self.encode_text(negative_prompt)
            negative_prompt_embeds = encoded_negative_text['last_hidden_state']
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            negative_prompt_mask = encoded_negative_text['attention_mask']
            negative_prompt_mask = negative_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
            negative_prompt_mask = (negative_prompt_mask == 1).to(device)
        else:
            negative_prompt_embeds = None

        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet_model_config["in_channels"]
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)
    

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress, leave=False)
        
        guidance_scale = self.unet_model_config['infer_classifier_free_guidance_strength'] if guidance_scale is None else guidance_scale

        all_latents = []

        for i, t in enumerate(timesteps):
            latent_model_input = latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            # expand t to batch size
            bsz = latent_model_input.shape[0]
            time = torch.full((bsz,), t, dtype=torch.long, device=device)

            noise_pred = self.unet(
                latent_model_input, time = time, embedding=prompt_embeds, embedding_scale=guidance_scale, embedding_mask_proba=0, negative_embedding = negative_prompt_embeds
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

            all_latents.append(latents)

        all_latents = torch.stack(all_latents, dim=0)
        if return_all_latents:
            return latents, all_latents
        else:
            return latents
    
    
    @torch.no_grad()
    def invert(
        self,
        start_latents,
        original_prompt = '',
        edit_prompt = '',
        guidance_scale=3,
        invert_steps=20,
        inference_steps=50,
        num_images_per_prompt=1,
        negative_prompt = None,
        return_intermediate_latents = False,
        verbose= False
    ):
    
        inversion_scheduler = DDIMInverseScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler", prediction_type=self.noise_scheduler.config.prediction_type)
        inversion_scheduler.config.prediction_type = self.noise_scheduler.config.prediction_type # to ensure the same prediction type
        print(f"Inversion scheduler initialized with {inversion_scheduler.config}") if verbose else None
    
    
        device = next(self.parameters()).device
        
        edit_prompt_embedding = self.encode_text(edit_prompt)[
            "last_hidden_state"
        ].repeat_interleave(num_images_per_prompt, 0)
        
        if negative_prompt:
            negative_prompt_embeddings = self.encode_text(negative_prompt)[
                "last_hidden_state"
            ].repeat_interleave(num_images_per_prompt, 0)
        else:
            negative_prompt_embeddings = None
            
        original_prompt_embeddings = self.encode_text(original_prompt)[
            "last_hidden_state"
        ].repeat_interleave(num_images_per_prompt, 0)
    
        latents = start_latents.to(device)
        intermediate_latents = [latents]
        
        inversion_scheduler.set_timesteps(inference_steps, device=device)
        self.inference_scheduler.set_timesteps(inference_steps, device=device)
        
        
        # print(f'inversion scheduler timestep: {inversion_scheduler.timesteps}')
        # print(f'inference scheduler timestep: {self.inference_scheduler.timesteps}')
        
        guidance_scale = self.unet_model_config['infer_classifier_free_guidance_strength'] if guidance_scale is None else guidance_scale

        ## go back to start steps using the inverse scheduler and then back to the start_latents with the new prompt
        
        ts = []
        for i in tqdm(range(invert_steps), disable= not verbose, leave = False, desc="Inverting"):
            
            t = inversion_scheduler.timesteps[i]
            ts.append(t)
            latent_model_input = latents
            latent_model_input = inversion_scheduler.scale_model_input(latent_model_input, t)
            
            time = torch.full((latents.shape[0],), t, dtype=torch.long, device=device)
            
            noise_pred = self.unet(
                latent_model_input, time = time, embedding=original_prompt_embeddings, embedding_scale=guidance_scale, embedding_mask_proba=0, negative_embedding = None
            )
            
            latents = inversion_scheduler.step(noise_pred, t, latents).prev_sample
            intermediate_latents.append(latents)
        
        edit_latents = latents.clone()
        latent_model_input = edit_latents
        intermediate_edit_latents = []
        
        # self.inference_scheduler.set_timesteps(invert_steps, device=device)
        
        # for i in tqdm(range(inference_steps- invert_steps, inference_steps), disable= not verbose, leave=False, desc="Reverting with edit"):
        
        for i,t in tqdm(enumerate(ts[::-1]), disable= not verbose, leave=False, desc="Reverting with edit"):
        
            # t = self.inference_scheduler.timesteps[i]
            
            latent_model_input = self.inference_scheduler.scale_model_input(latent_model_input, t)
            
            bsz = latent_model_input.shape[0]
            time = torch.full((bsz,), t, dtype=torch.long, device=device)
            
            pred = self.unet(
                latent_model_input, time = time, embedding=edit_prompt_embedding, embedding_scale=guidance_scale, embedding_mask_proba=0, negative_embedding = negative_prompt_embeddings
            )
            
            latents = self.inference_scheduler.step(pred, t, latents).prev_sample
            intermediate_edit_latents.append(latents)
            latent_model_input = latents
            
        intermediate_edit_latents = torch.stack(intermediate_edit_latents, dim=0)
        intermediate_latents = torch.stack(intermediate_latents, dim=0)
        
        return intermediate_edit_latents[-1], intermediate_latents, intermediate_edit_latents if return_intermediate_latents else intermediate_edit_latents[-1]
    
    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 64)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents
    
    
class LightningDiffGar(DiffGarLDM,LightningModule):
    
    def __init__(self,
                encoder_pair='clap',
                encoder_pair_kwargs=None,
                encoder_pair_ckpt=None,
                text_encoder = None,
                text_encoder_kwargs = None,
                text_encoder_ckpt = None,
                scheduler_name='stabilityai/stable-diffusion-2-1',
                scheduler_pred_type='epsilon',
                unet_model_config=None,
                unet_ckpt=None,
                freeze_encoder_pair=True,
                freeze_unet=False,
                contrastive_loss = False,
                contrastive_loss_kwargs = {
                    'temperature': 0.1,
                    'weight': 0.5,
                },
                preextracted_latents = True,
                optimizer: OptimizerCallable = None,
                scheduler = None,
                retrieval_every_n_epochs = 50,
                **kwargs
                ):
        
        DiffGarLDM.__init__(self,
                    encoder_pair=encoder_pair,
                    encoder_pair_kwargs=encoder_pair_kwargs,
                    encoder_pair_ckpt=encoder_pair_ckpt,
                    scheduler_name=scheduler_name,
                    text_encoder=text_encoder,
                    text_encoder_kwargs=text_encoder_kwargs,
                    text_encoder_ckpt=text_encoder_ckpt,
                    scheduler_pred_type=scheduler_pred_type,
                    unet_model_config=unet_model_config,
                    unet_ckpt=unet_ckpt,
                    freeze_encoder_pair=freeze_encoder_pair,
                    freeze_unet=freeze_unet,
                    contrastive_loss=contrastive_loss,
                    contrastive_loss_kwargs=contrastive_loss_kwargs)
        
        self.preextracted_latents = preextracted_latents
        self.gen_examples = True
        self.first_run = True
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.val_audio_preds = [[] for _ in range(10)]
        self.val_gt_audio = [[] for _ in range(10)]
        self.val_gt_text = [[] for _ in range(10)]
        
        self.train_audio_preds = []
        self.train_gt_audio = []
        self.train_gt_text = []
        
        self.retrieval_every_n_epochs = retrieval_every_n_epochs
        
        
    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        
        ## if the prompt length is below the unet model max length, randomly repeat the prompt
        for i, p_ in enumerate(prompt):
            if len(p_) < self.unet_model_config['embedding_max_length'] and np.random.rand() > 0.7:
                p_ = [p_]* (self.unet_model_config['embedding_max_length'] // len(p_))
                p_ = '. '.join(p_)
                prompt[i] = p_
                
        
        b = audio.shape[0]
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt)
        
        if latents.dtype == torch.float16:
            latents = latents.float()
        
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for key in loss_dict:
            self.log(f'train_{key}', loss_dict[key], on_step=True, on_epoch=True, prog_bar=True)
        self.log('contrastive_loss_weight', self.contrastive_loss.weight, on_step=True, on_epoch=False, prog_bar=True) if self.contrastive_loss else None
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        
        if self.current_epoch % self.retrieval_every_n_epochs == 0 and self.gen_examples and len(self.train_audio_preds) < self.trainer.limit_val_batches:    
            print(f"Generating some samples")
            
            preds = self.inference(prompt, self.inference_scheduler, num_steps = 50, disable_progress = False, guidance_scale = self.unet_model_config['infer_classifier_free_guidance_strength'])
            
            preds = preds.permute(0,2,1)
            
            
            print(f'Generated samples of shape {preds.shape}') if self.first_run else None
            print(f'Ground truth samples of shape {latents.shape}') if self.first_run else None
            
            try: 
                gt_clap = self.encoder_pair.get_clap_score(latents, prompt, latents = True)['CLAP_Score']
                pred_clap = self.encoder_pair.get_clap_score(preds, prompt, latents = True)['CLAP_Score']
                # print(f"Computing CLAP score") if self.first_run else None
                # print(f"Ground truth CLAP score: {gt_clap}") if self.first_run else None
                # print(f"Computing CLAP score for generated samples") if self.first_run else None
                # print(f"Generated CLAP score: {pred_clap}") if self.first_run else None
                self.log('gt_clap', gt_clap, on_step=True, on_epoch=True, prog_bar=True)
                self.log('pred_clap', pred_clap, on_step=True, on_epoch=True, prog_bar=True)
            except:
                pass
            
            norm_preds = preds.mean(dim=1) / preds.mean(dim=1).norm(dim=1, keepdim=True)
            norm_latents = latents.mean(dim=1) / latents.mean(dim=1).norm(dim=1, keepdim=True)
            
            audio_to_audio_sims = norm_preds @ norm_latents.t()
            audio_to_audio_sims = audio_to_audio_sims.diag().mean()
            gt_audio_to_audio_sims = norm_latents @ norm_latents.t()
            gt_audio_to_audio_sims = gt_audio_to_audio_sims.diag().mean()
            
            self.log('A2A_CLAP', audio_to_audio_sims, on_step=True, on_epoch=True, prog_bar=True)
            self.log('GT_A2A_CLAP', gt_audio_to_audio_sims, on_step=True, on_epoch=True, prog_bar=True)
            self.train_gt_audio.append(latents.mean(dim=1).detach().cpu())
            self.train_audio_preds.append(preds.mean(dim=1).detach().cpu())
            self.train_gt_text.append(self.encode_text(prompt).get('projected_pooler_output', self.encode_text(prompt)['last_hidden_state'].mean(1)).detach().cpu())
            
            
            
        if self.scheduler is not None:
            self.scheduler.step()
        return loss
        
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        
        
        audio = batch['audio']
        prompt = batch['prompt']
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt, validation_mode=True)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for key in loss_dict:
            self.log(f'val_{key}', loss_dict[key], on_step=False, on_epoch=True, prog_bar=True)
            
            
        ## generate some samples for validation
        if self.current_epoch % self.retrieval_every_n_epochs == 0:
            preds = self.inference(prompt, self.inference_scheduler, num_steps = 50, disable_progress = False, guidance_scale = self.unet_model_config['infer_classifier_free_guidance_strength'])
            preds = preds.permute(0,2,1)
            preds = preds.mean(dim=1)
            latents = latents.mean(dim=1)
            text_dict = self.encode_text(prompt)
            text_embedding = text_dict.get('projected_pooler_output', text_dict['last_hidden_state'].mean(1))
            
            self.val_audio_preds[dataloader_idx].append(preds.detach().cpu())
            self.val_gt_audio[dataloader_idx].append(latents.detach().cpu())
            self.val_gt_text[dataloader_idx].append(text_embedding.detach().cpu())
            
        
        return loss
    
        
    def configure_optimizers(self):
        return configure_optimizers_(self_=self)
    
    
    def on_validation_epoch_end(self, thresholds = [1,5,10]):
        on_validation_epoch_end_(self, thresholds)
        
    def on_train_epoch_end(self, thresholds = [1,5,10]):
        on_train_epoch_end_(self, thresholds)
        
    
    
    
class MSEGar(BaseModule):
    def __init__(
        self,
        encoder_pair='clap', ## this will be used for clap score, but the text encoder can be different
        encoder_pair_kwargs=None,
        encoder_pair_ckpt=None,
        text_encoder = None,
        text_encoder_kwargs = None,
        text_encoder_ckpt = None,
        unet_model_config=None,
        unet_ckpt=None,
        freeze_encoder_pair=True,
        freeze_unet=False,
        device=None,
        **kwargs
        
    ):
        super().__init__()

        self.unet_model_config = unet_model_config
        self.max_length = self.unet_model_config['embedding_max_length'] if self.unet_model_config is not None else 0
        assert self.unet_model_config is not None, "UNet model config is required"
                
        self.unet = UNet.from_config(self.unet_model_config) if self.unet_model_config is not None else None
        self.rand_token = nn.parameter.Parameter(torch.randn(1,self.unet_model_config['in_channels']))   

        self.encoder_pair = get_encoder_pair(encoder_pair, encoder_pair_kwargs)
        if encoder_pair_ckpt:
            self.encoder_pair.load_ckpt(encoder_pair_ckpt, verbose=False)
        
        self.text_encoder = None
        if text_encoder is not None:
            self.text_encoder = get_text_encoder(text_encoder, text_encoder_kwargs)
            self.text_encoder.max_length = self.max_length
            if text_encoder_ckpt:
                self.text_encoder.load_ckpt(text_encoder_ckpt)
        
        if freeze_encoder_pair:
            self.freeze_encoder_pair()
            self.freeze_text_encoder()
            
        self.freeze_encoders = freeze_encoder_pair
        
        if unet_ckpt:
            ckpt = torch.load(unet_ckpt)
            self.unet.load_state_dict(ckpt['state_dict'])
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
        
        self.first_run = False
        
        
        with torch.no_grad():
            text_dim = self.text_encoder.get_text_embedding("test")['last_hidden_state'].shape[-1] if self.text_encoder is not None else self.encoder_pair.get_text_embedding("test")['last_hidden_state'].shape[-1]
        
        print(f"Text dimension: {text_dim}")
        
        if device is not None:
            self.to(device)
     
              
       
    def forward(self, latents, prompt, validation_mode=False):
        
        latents = latents.float()
        text_dict = self.encode_text(prompt)
        text_embedding = text_dict['last_hidden_state']
        ## tile the text embedding to match the latents, latents are of shape (bsz, channels, length) and text_embedding is of shape (bsz, channels)
        
        target = latents
        input_ = self.rand_token.unsqueeze(-1).expand(latents.shape[0],-1,latents.shape[-1])
        timesteps = torch.zeros((latents.shape[0],), dtype=torch.int64, device=latents.device)
        
        model_pred = self.unet(
            input_, time = timesteps, embedding = text_embedding, embedding_mask_proba = 0, embedding_scale = 1.0
        )
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        mse_loss = loss
            
        loss_dict = {
            'mse_loss': mse_loss,
        }

        assert model_pred.shape == target.shape, "Model prediction and target shape mismatch"

        return loss, loss_dict

    @torch.no_grad()
    def inference(self, prompt,num_samples_per_prompt=1, return_all_latents = False, **kwargs):
        device = next(self.parameters()).device
        batch_size = len(prompt) * num_samples_per_prompt
        
        
        encoded_text = self.encode_text(prompt)
        text_embedding = encoded_text['last_hidden_state']
        num_channels_latents = self.unet_model_config["in_channels"]
        latents = self.prepare_latents(batch_size, num_channels_latents, text_embedding.dtype, device)
        input_ = self.rand_token.unsqueeze(-1).expand(latents.shape[0],-1,latents.shape[-1])
        timesteps = torch.zeros((latents.shape[0],), dtype=torch.int64, device=latents.device)


        pred = self.unet(
            input_,
            time = timesteps,
            embedding = text_embedding,
            embedding_mask_proba = 0,
            embedding_scale = 1.0
        )

        if return_all_latents:
            return pred, None
        else:
            return pred
    

    def prepare_latents(self, batch_size, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 64)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        return latents

class LightningMSEGar(MSEGar,LightningModule):
    
    def __init__(self,
                encoder_pair='clap',
                encoder_pair_kwargs=None,
                encoder_pair_ckpt=None,
                text_encoder = None,
                text_encoder_kwargs = None,
                text_encoder_ckpt = None,
                unet_model_config=None,
                unet_ckpt=None,
                freeze_encoder_pair=True,
                freeze_unet=False,
                contrastive_loss = False,
                preextracted_latents = True,
                optimizer: OptimizerCallable = None,
                scheduler = None,
                scheduler_name='stabilityai/stable-diffusion-2-1',
                scheduler_pred_type='epsilon',
                retrieval_every_n_epochs = 50,
                **kwargs
                ):
        
        MSEGar.__init__(self,
                        encoder_pair=encoder_pair,
                        encoder_pair_kwargs=encoder_pair_kwargs,
                        encoder_pair_ckpt=encoder_pair_ckpt,
                        text_encoder=text_encoder,
                        text_encoder_kwargs=text_encoder_kwargs,
                        text_encoder_ckpt=text_encoder_ckpt,
                        unet_model_config=unet_model_config,
                        unet_ckpt=unet_ckpt,
                        freeze_encoder_pair=freeze_encoder_pair,
                        freeze_unet=freeze_unet,
                        contrastive_loss=contrastive_loss
                        )
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.preextracted_latents = preextracted_latents
        self.gen_examples = True
        self.first_run = True
        self.retrieval_every_n_epochs = retrieval_every_n_epochs
        
        self.val_audio_preds = [[] for _ in range(10)]
        self.val_gt_audio = [[] for _ in range(10)]
        self.val_gt_text = [[] for _ in range(10)]
        
        self.train_audio_preds = []
        self.train_gt_audio = []
        self.train_gt_text = []
        self.train_prompts = []
        
        
    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        file_path  = batch['file_path']
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        latents = latents.permute(0,2,1).float()
        loss, loss_dict = self(latents, prompt)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        for key in loss_dict:
            self.log(f'train_{key}', loss_dict[key], on_step=True, on_epoch=True, prog_bar=True)
        
        # log the learning rate
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        
        if self.current_epoch % self.retrieval_every_n_epochs == 0 and self.gen_examples and len(self.train_audio_preds) <  self.trainer.limit_val_batches:
            print(f"Generating some samples")
            preds = self.inference(prompt)
            
            preds,latents = preds.permute(0,2,1), latents.permute(0,2,1)
            
            print(f'Generated samples of shape {preds.shape}') if self.first_run else None
            print(f'Ground truth samples of shape {latents.shape}') if self.first_run else None
            
            try: 
                gt_clap = self.encoder_pair.get_clap_score(latents, prompt, latents = True)['CLAP_Score']
                pred_clap = self.encoder_pair.get_clap_score(preds, prompt, latents = True)['CLAP_Score']
                print(f"Computing CLAP score") if self.first_run else None
                print(f"Ground truth CLAP score: {gt_clap}") if self.first_run else None
                print(f"Computing CLAP score for generated samples") if self.first_run else None
                print(f"Generated CLAP score: {pred_clap}") if self.first_run else None
                self.log('gt_clap', gt_clap, on_step=True, on_epoch=True, prog_bar=True)
                self.log('pred_clap', pred_clap, on_step=True, on_epoch=True, prog_bar=True)
            
            except Exception as e:
                print(f"Error computing CLAP score: {e}")
                
                
            norm_preds = preds.mean(dim=1) / preds.mean(dim=1).norm(dim=1, keepdim=True)
            norm_latents = latents.mean(dim=1) / latents.mean(dim=1).norm(dim=1, keepdim=True)
                
            
            audio_to_audio_sims = norm_preds @ norm_latents.t()
            audio_to_audio_sims = audio_to_audio_sims.diag().mean()
            gt_audio_to_audio_sims = norm_latents @ norm_latents.t()
            gt_audio_to_audio_sims = gt_audio_to_audio_sims.diag().mean()
            
            self.log('A2A_CLAP', audio_to_audio_sims, on_step=True, on_epoch=True, prog_bar=True)
            self.log('GT_A2A_CLAP', gt_audio_to_audio_sims, on_step=True, on_epoch=True, prog_bar=True)
            
            self.train_gt_audio.append(latents.mean(dim=1).detach().cpu())
            self.train_audio_preds.append(preds.mean(dim=1).detach().cpu())
            self.train_gt_text.append(self.encode_text(prompt).get('projected_pooler_output', self.encode_text(prompt)['last_hidden_state'].mean(1)).detach().cpu())
            
            
        if self.scheduler is not None:
            self.scheduler.step()
        return loss
        
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt, validation_mode=True)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        for key in loss_dict:
            self.log(f'val_{key}', loss_dict[key], on_step=False, on_epoch=True, prog_bar=True)
        
        
        if self.current_epoch % self.retrieval_every_n_epochs == 0:
            preds = self.inference(prompt)
            preds = preds.permute(0,2,1)
            preds = preds.mean(dim=1)
            latents = latents.mean(dim=1)
            text_dict = self.encode_text(prompt)
            text_embedding = text_dict.get('projected_pooler_output', text_dict['last_hidden_state'].mean(1))
            
            self.val_audio_preds[dataloader_idx].append(preds.detach().cpu())
            self.val_gt_audio[dataloader_idx].append(latents.detach().cpu())
            self.val_gt_text[dataloader_idx].append(text_embedding.detach().cpu())
        
        
        return loss
    
    
    
    def configure_optimizers(self):
        return configure_optimizers_(self_=self)
    
    
    def on_validation_epoch_end(self, thresholds = [1,5,10]):
        on_validation_epoch_end_(self, thresholds)
        
    def on_train_epoch_end(self, thresholds = [1,5,10]):
        on_train_epoch_end_(self, thresholds)
        
        
    
class MLPldm(BaseModule):
    def __init__(
        self,
        encoder_pair='clap', ## this will be used for clap score, but the text encoder can be different
        encoder_pair_kwargs=None,
        encoder_pair_ckpt=None,
        text_encoder = None,
        text_encoder_kwargs = None,
        text_encoder_ckpt = None,
        mlp_model_config=None,
        mlp_ckpt=None,
        freeze_encoder_pair=True,
        freeze_mlp=False,
        device=None,
        scheduler_name='stabilityai/stable-diffusion-2-1',
        scheduler_pred_type='epsilon',  
        **kwargs
    ):
        
        from diffgar.models.ldm.mlp import MLP
        
        ''' this is the same as DiffgarLDM but with a MLP instead of a UNet, and adaLN instead of cross-attention'''
    
        super().__init__()


        self.scheduler_name = scheduler_name
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.inference_scheduler = DDIMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.noise_scheduler.config.prediction_type = scheduler_pred_type
        self.inference_scheduler.config.prediction_type = scheduler_pred_type
        self.noise_scheduler.register_to_config(prediction_type=scheduler_pred_type)
        self.inference_scheduler.register_to_config(prediction_type=scheduler_pred_type)
        
        
        self.mlp_model_config = mlp_model_config
        self.max_length = self.mlp_model_config['embedding_max_length'] if self.mlp_model_config is not None else 0
        self.mlp = MLP.from_config(self.mlp_model_config) if self.mlp_model_config is not None else MLP()
        print("MLP initialized randomly.")

        self.encoder_pair = get_encoder_pair(encoder_pair, encoder_pair_kwargs)
        if encoder_pair_ckpt:
            self.encoder_pair.load_ckpt(encoder_pair_ckpt, verbose=False)
        
        self.text_encoder = None
        if text_encoder is not None:
            self.text_encoder = get_text_encoder(text_encoder, text_encoder_kwargs)
            self.text_encoder.max_length = self.max_length
            if text_encoder_ckpt:
                self.text_encoder.load_ckpt(text_encoder_ckpt)
        
        if freeze_encoder_pair:
            self.freeze_encoder_pair()
            self.freeze_text_encoder()
            
        self.freeze_encoders = freeze_encoder_pair
        
        if mlp_ckpt:
            ckpt = torch.load(mlp_ckpt)
            self.mlp.load_state_dict(ckpt['state_dict'])
            
        if freeze_mlp:
            for param in self.mlp.parameters():
                param.requires_grad = False
                
        self.first_run = False
        
        ## create a mask token for cfg 
        self.rand_token = self.text_encoder.get_text_embedding([""]) if self.text_encoder is not None else self.encoder_pair.get_text_embedding("")
        self.rand_token = self.rand_token.get('projected_pooler_output', self.rand_token['last_hidden_state'].mean(dim=1))
        
        
        with torch.no_grad():
            enc_test_ = self.text_encoder.get_text_embedding("test") if self.text_encoder is not None else self.encoder_pair.get_text_embedding("test")
            text_dim = enc_test_.get('projected_pooler_output', enc_test_['last_hidden_state']).shape[-1]
        
        print(f"Text dimension: {text_dim}")
        
        if device is not None:
            self.to(device)
        
    def forward(self, latents, prompt, validation_mode=False, embedding_mask_proba = None, flip = False):
        
        
        #if half precision, to float
        if latents.dtype == torch.float16:
            latents = latents.float()
        
        bsz, _ = latents.shape # B, d
        device = latents.device
        
                
        if self.mlp_model_config['classifier_free_guidance_strength'] is not None:
            guidance_scale = self.mlp_model_config['classifier_free_guidance_strength']
        if embedding_mask_proba is not None:
            guidance_scale = embedding_mask_proba        
        # create a cfg mask
        mask = torch.bernoulli(torch.full((bsz, 1), guidance_scale, device=device)).bool()
        # prompt = [p if m else "" for p, m in zip(prompt, mask.squeeze().tolist())]
        text_dict = self.encode_text(prompt)    
        conditioning = text_dict.get('projected_pooler_output', text_dict['last_hidden_state'].mean(dim=1)).to(device)
    
        
        if flip:
            conditioning, latents = latents, conditioning
            
        # apply rand token to mask
        conditioning = torch.where(mask, conditioning, self.rand_token.to(device))
        
            
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)
        
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device).long()
                
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = latents
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        assert latents.shape == noisy_latents.shape, "Latents and noisy latents shape mismatch"
        assert latents.shape == target.shape, "Latents and target shape mismatch"

        model_pred = self.mlp(
            x = noisy_latents,
            conditioning = conditioning,
            timesteps = timesteps,
        )

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
        mse_loss = loss

        loss_dict = {
            'mse_loss': mse_loss,
        }
    
        assert model_pred.shape == target.shape, "Model prediction and target shape mismatch"
        return loss, loss_dict

    @torch.no_grad()
    def inference(self, prompt, inference_scheduler = None, num_steps=20, guidance_scale=None, num_samples_per_prompt=1, 
                disable_progress=True, return_all_latents = False, negative_prompt = None, flip = False, **kwargs):
        device = next(self.parameters()).device
        batch_size = len(prompt) * num_samples_per_prompt
        #TODO deal with audio latents
        
        inference_scheduler = self.inference_scheduler if inference_scheduler is None else inference_scheduler 
        
        encoded_text = self.encode_text(prompt)
        prompt_embeds = encoded_text.get('projected_pooler_output', encoded_text['last_hidden_state'].mean(dim=1))
        boolean_prompt_mask = encoded_text['attention_mask']
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
        boolean_prompt_mask = (boolean_prompt_mask == 1).to(device)
        
        num_channels_latents = self.mlp_model_config["in_channels"]
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)        
        guidance_scale = self.mlp_model_config['infer_classifier_free_guidance_strength'] if guidance_scale is None else guidance_scale

        
        #unconditional embeddings for classifier free guidance
        if guidance_scale is not None:
            if negative_prompt is None:
                negative_prompt_embeds = self.rand_token.repeat_interleave(len(prompt), 0)
            else:
                uncond_encoded_text_dict = self.encode_text(negative_prompt)
                negative_prompt_embeds = uncond_encoded_text_dict.get('projected_pooler_output', uncond_encoded_text_dict['last_hidden_state'].mean(dim=1))
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)

            print(f"Negative prompt embeddings shape: {negative_prompt_embeds.shape}")
            print(f"Prompt embeddings shape: {prompt_embeds.shape}")

            prompt_embeds = torch.cat([negative_prompt_embeds.to(device), prompt_embeds.to(device)], dim=0)

        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)
        
        all_latents = []

        for i, t in enumerate(timesteps):
            
            
            latents = torch.cat([latents, latents]) if guidance_scale is not None else latents
            latent_model_input = latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            # expand t to batch size
            bsz = latent_model_input.shape[0]
            time = torch.full((bsz,), t, dtype=torch.long, device=device)

            noise_pred = self.mlp(
                x = latent_model_input,
                conditioning = prompt_embeds,
                timesteps = time)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample
            
            #interpolate the latents for the classifier free guidance
            if guidance_scale is not None:
                unconditional_latents, conditional_latents = torch.chunk(latents, 2, dim=0)
                latents = unconditional_latents + (conditional_latents - unconditional_latents) * guidance_scale

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)
                
            all_latents.append(latents.unsqueeze(-1).expand(-1, -1, 64))

        all_latents = torch.stack(all_latents, dim=0)
        
        
        if return_all_latents:
            return latents.unsqueeze(-1).expand(-1, -1, 64), all_latents
        else:
            return latents.unsqueeze(-1).expand(-1, -1, 64)

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents


class LightningMLPldm(MLPldm,LightningModule):
    
    def __init__(self,
                encoder_pair='clap',
                encoder_pair_kwargs=None,
                encoder_pair_ckpt=None,
                text_encoder = None,
                text_encoder_kwargs = None,
                text_encoder_ckpt = None,
                mlp_model_config=None,
                mlp_ckpt=None,
                snr_gamma=None,
                freeze_encoder_pair=True,
                freeze_mlp=False,
                optimizer: OptimizerCallable = None,
                scheduler = None,
                scheduler_name='stabilityai/stable-diffusion-2-1',
                scheduler_pred_type='epsilon',
                preextracted_latents = True,
                bidirectional_interval=None,
                retrieval_every_n_epochs = 50,
                **kwargs
                ):
        
        MLPldm.__init__(self,
                    encoder_pair=encoder_pair,
                    encoder_pair_kwargs=encoder_pair_kwargs,
                    encoder_pair_ckpt=encoder_pair_ckpt,
                    text_encoder=text_encoder,
                    text_encoder_kwargs=text_encoder_kwargs,
                    text_encoder_ckpt=text_encoder_ckpt,
                    mlp_model_config=mlp_model_config,
                    mlp_ckpt=mlp_ckpt,
                    snr_gamma=snr_gamma,
                    freeze_encoder_pair=freeze_encoder_pair,
                    freeze_mlp=freeze_mlp,
                    scheduler_name=scheduler_name,
                    scheduler_pred_type=scheduler_pred_type
                    )
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.gen_examples = True
        self.first_run = True
        self.preextracted_latents = preextracted_latents
        self.bidirectional_interval = bidirectional_interval
        self.flip = False
        self.retrieval_every_n_epochs = retrieval_every_n_epochs
        
        self.val_audio_preds = [[] for _ in range(10)]
        self.val_gt_audio = [[] for _ in range(10)]
        self.val_gt_text = [[] for _ in range(10)]
        
        self.train_audio_preds = []
        self.train_gt_audio = []
        self.train_gt_text = []
        self.train_prompts = []
        
        
    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        latents_in = latents.mean(1)
        
        if self.bidirectional_interval is not None:
            if self.global_step % self.bidirectional_interval == 0:
                self.flip = not self.flip
                print(f"==================== Flipping the bidirectional training ====================")
        
        if not self.flip:
            loss, loss_dict = self(latents_in, prompt)
        else:
            loss, loss_dict = self(latents_in, prompt, flip = True)
        
        
        if latents.dtype == torch.float16:
            latents = latents.float()
        
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        for key in loss_dict:
            self.log(f'train_{key}', loss_dict[key], on_step=True, on_epoch=True, prog_bar=True)
        
        # log the learning rate
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        
        if self.current_epoch % self.retrieval_every_n_epochs == 0 and self.gen_examples and len(self.train_audio_preds) < self.trainer.limit_val_batches:
            print(f"Generating some samples")
            preds = self.inference(prompt, num_steps = 50, disable_progress = False, guidance_scale = self.mlp_model_config['infer_classifier_free_guidance_strength'])
            preds = preds.permute(0,2,1)
            
            print(f'Generated samples of shape {preds.shape}') if self.first_run else None
            print(f'Ground truth samples of shape {latents.shape}') if self.first_run else None
            
            
            
            try: 
                gt_clap = self.encoder_pair.get_clap_score(latents, prompt, latents = True)['CLAP_Score']
                pred_clap = self.encoder_pair.get_clap_score(preds, prompt, latents = True)['CLAP_Score']
                print(f"Computing CLAP score") if self.first_run else None
                print(f"Ground truth CLAP score: {gt_clap}") if self.first_run else None
                print(f"Computing CLAP score for generated samples") if self.first_run else None
                print(f"Generated CLAP score: {pred_clap}") if self.first_run else None
                self.log('gt_clap', gt_clap, on_step=True, on_epoch=True, prog_bar=True)
                self.log('pred_clap', pred_clap, on_step=True, on_epoch=True, prog_bar=True)
                
            except Exception as e:
                print(f"Error computing CLAP score: {e}")
                
                
            norm_preds = preds.mean(dim=1)/torch.norm(preds.mean(dim=1), dim=1).unsqueeze(1)
            norm_latents = latents.mean(dim=1)/torch.norm(latents.mean(dim=1), dim=1).unsqueeze(1)    
            
            audio_to_audio_sims = norm_preds @ norm_latents.t()
            audio_to_audio_sims = audio_to_audio_sims.diag().mean()
            gt_audio_to_audio_sims = norm_latents @ norm_latents.t()
            gt_audio_to_audio_sims = gt_audio_to_audio_sims.diag().mean()
            
            self.log('A2A_CLAP', audio_to_audio_sims, on_step=True, on_epoch=True, prog_bar=True)
            self.log('GT_A2A_CLAP', gt_audio_to_audio_sims, on_step=True, on_epoch=True, prog_bar=True)
            
            self.train_audio_preds.append(preds.mean(dim=1).detach().cpu())
            self.train_gt_audio.append(latents.mean(dim=1).detach().cpu())
            self.train_gt_text.append(self.encode_text(prompt).get('projected_pooler_output', self.encode_text(prompt)['last_hidden_state'].mean(1)).detach().cpu())
            
            
        if self.scheduler is not None:
            self.scheduler.step()
        return loss
    
    
    def on_validation_epoch_end(self, thresholds = [1,5,10]):
        on_validation_epoch_end_(self, thresholds)
        
    def on_train_epoch_end(self, thresholds = [1,5,10]):
        on_train_epoch_end_(self, thresholds)
        
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        loss, loss_dict = self(latents.mean(1), prompt, validation_mode=True)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        for key in loss_dict:
            self.log(f'val_{key}', loss_dict[key], on_step=False, on_epoch=True, prog_bar=True)
            
        ## generate some samples for validation
        if self.current_epoch % self.retrieval_every_n_epochs == 0:
            preds = self.inference(prompt, self.inference_scheduler, num_steps = 50, disable_progress = False, guidance_scale = self.mlp_model_config['infer_classifier_free_guidance_strength'])
            preds = preds.permute(0,2,1)
            preds = preds.mean(dim=1)
            latents = latents.mean(dim=1)
            text_dict = self.encode_text(prompt)
            text_embedding = text_dict.get('projected_pooler_output', text_dict['last_hidden_state'].mean(1))
            
            self.val_audio_preds[dataloader_idx].append(preds.detach().cpu())
            self.val_gt_audio[dataloader_idx].append(latents.detach().cpu())
            self.val_gt_text[dataloader_idx].append(text_embedding.detach().cpu())
            
        return loss
    
    
    def configure_optimizers(self):
        return configure_optimizers_(self_=self)
        
        

class MLPMSE(BaseModule):
    def __init__(
        self,
        encoder_pair='clap', ## this will be used for clap score, but the text encoder can be different
        encoder_pair_kwargs=None,
        encoder_pair_ckpt=None,
        text_encoder = None,
        text_encoder_kwargs = None,
        text_encoder_ckpt = None,
        mlp_model_config=None,
        mlp_ckpt=None,
        freeze_encoder_pair=True,
        freeze_mlp=False,
        device=None,
        scheduler_name='stabilityai/stable-diffusion-2-1',
        scheduler_pred_type='epsilon',  
        **kwargs
    ):
        
        from diffgar.models.ldm.mlp import MLP
        
        ''' this is the same as DiffgarLDM but with a MLP instead of a UNet, and adaLN instead of cross-attention'''
    
        super().__init__()


        
        
        self.mlp_model_config = mlp_model_config
        self.max_length = self.mlp_model_config['embedding_max_length'] if self.mlp_model_config is not None else 0
        self.mlp = MLP.from_config(self.mlp_model_config) if self.mlp_model_config is not None else MLP()
        print("MLP initialized randomly.")

        self.encoder_pair = get_encoder_pair(encoder_pair, encoder_pair_kwargs)
        if encoder_pair_ckpt:
            self.encoder_pair.load_ckpt(encoder_pair_ckpt, verbose=False)
        
        self.text_encoder = None
        if text_encoder is not None:
            self.text_encoder = get_text_encoder(text_encoder, text_encoder_kwargs)
            self.text_encoder.max_length = self.max_length
            if text_encoder_ckpt:
                self.text_encoder.load_ckpt(text_encoder_ckpt)
        
        if freeze_encoder_pair:
            self.freeze_encoder_pair()
            self.freeze_text_encoder()
            
        self.freeze_encoders = freeze_encoder_pair
        
        if mlp_ckpt:
            ckpt = torch.load(mlp_ckpt)
            self.mlp.load_state_dict(ckpt['state_dict'])
            
        if freeze_mlp:
            for param in self.mlp.parameters():
                param.requires_grad = False
                
        self.first_run = False
        self.rand_token = nn.parameter.Parameter(torch.randn(1,self.mlp_model_config['in_channels']))
        
        
        with torch.no_grad():
            enc_test_ = self.text_encoder.get_text_embedding("test") if self.text_encoder is not None else self.encoder_pair.get_text_embedding("test")
            text_dim = enc_test_.get('projected_pooler_output', enc_test_['last_hidden_state']).shape[-1]
        
        print(f"Text dimension: {text_dim}")
        
        if device is not None:
            self.to(device)
        
    def forward(self, latents, prompt, validation_mode=False, embedding_mask_proba = None):
        
        
        #if half precision, to float
        if latents.dtype == torch.float16:
            latents = latents.float()
        
        bsz, _ = latents.shape # B, d
        device = latents.device
        
        
        timesteps = torch.zeros((bsz,), dtype=torch.int64, device=latents.device)
        text_dict = self.encode_text(prompt)    
        encoder_hidden_states = text_dict.get('projected_pooler_output', text_dict['last_hidden_state'].mean(1))
        input_ = self.rand_token.expand(latents.shape[0],-1)
    
        target = latents
    
        model_pred = self.mlp(
            x = input_,
            conditioning = encoder_hidden_states,
            timesteps = timesteps,
        )

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
        mse_loss = loss

        loss_dict = {
            'mse_loss': mse_loss,
        }
    
        assert model_pred.shape == target.shape, "Model prediction and target shape mismatch"
        return loss, loss_dict

    @torch.no_grad()
    def inference(self, prompt, inference_scheduler = None, num_samples_per_prompt=1,  return_all_latents = False, **kwargs):
        
        
        device = next(self.parameters()).device
        batch_size = len(prompt) * num_samples_per_prompt
        encoded_text = self.encode_text(prompt)
        prompt_embeds = encoded_text.get('projected_pooler_output', encoded_text['last_hidden_state'].mean(1))
        timesteps = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        input_ = self.rand_token.expand(batch_size,-1)
        
        preds = self.mlp(
            x = input_,
            conditioning = prompt_embeds,
            timesteps = timesteps,
        )
        
        
        
        if return_all_latents:
            return preds, None
        else:
            return preds.unsqueeze(-1).expand(-1, -1, 64)
        
    def prepare_latents(self, batch_size, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        return latents


class LightningMLPMSE(MLPMSE,LightningModule):
    
    def __init__(self,
                encoder_pair='clap',
                encoder_pair_kwargs=None,
                encoder_pair_ckpt=None,
                text_encoder = None,
                text_encoder_kwargs = None,
                text_encoder_ckpt = None,
                mlp_model_config=None,
                mlp_ckpt=None,
                snr_gamma=None,
                freeze_encoder_pair=True,
                freeze_mlp=False,
                optimizer: OptimizerCallable = None,
                scheduler = None,
                scheduler_name='stabilityai/stable-diffusion-2-1',
                scheduler_pred_type='epsilon',
                preextracted_latents = True,
                retrieval_every_n_epochs = 50,
                **kwargs
                ):
        
        MLPMSE.__init__(self,
                    encoder_pair=encoder_pair,
                    encoder_pair_kwargs=encoder_pair_kwargs,
                    encoder_pair_ckpt=encoder_pair_ckpt,
                    text_encoder=text_encoder,
                    text_encoder_kwargs=text_encoder_kwargs,
                    text_encoder_ckpt=text_encoder_ckpt,
                    mlp_model_config=mlp_model_config,
                    mlp_ckpt=mlp_ckpt,
                    snr_gamma=snr_gamma,
                    freeze_encoder_pair=freeze_encoder_pair,
                    freeze_mlp=freeze_mlp,
                    scheduler_name=scheduler_name,
                    scheduler_pred_type=scheduler_pred_type
                    )
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.gen_examples = True
        self.first_run = True
        self.preextracted_latents = preextracted_latents
        
        
        self.retrieval_every_n_epochs = retrieval_every_n_epochs
        
        self.val_audio_preds = [[] for _ in range(10)]
        self.val_gt_audio = [[] for _ in range(10)]
        self.val_gt_text = [[] for _ in range(10)]
        
        self.train_audio_preds = []
        self.train_gt_audio = []
        self.train_gt_text = []
        self.train_prompts = []
        
    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        loss, loss_dict = self(latents.mean(1), prompt)
        
        if latents.dtype == torch.float16:
            latents = latents.float()
        
        
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        for key in loss_dict:
            self.log(f'train_{key}', loss_dict[key], on_step=True, on_epoch=True, prog_bar=True)
        
        # log the learning rate
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        
        if self.current_epoch % self.retrieval_every_n_epochs == 0 and self.gen_examples and len(self.train_audio_preds) < self.trainer.limit_val_batches:
            print(f"Generating some samples")
            preds = self.inference(prompt)
            
            print(f'Generated samples of shape {preds.shape}') if self.first_run else None
            print(f'Ground truth samples of shape {latents.shape}') if self.first_run else None
            preds = preds.permute(0,2,1)
            
            try: 
                gt_clap = self.encoder_pair.get_clap_score(latents, prompt, latents = True)['CLAP_Score']
                pred_clap = self.encoder_pair.get_clap_score(preds, prompt, latents = True)['CLAP_Score']
                print(f"Computing CLAP score") if self.first_run else None
                print(f"Ground truth CLAP score: {gt_clap}") if self.first_run else None
                print(f"Computing CLAP score for generated samples") if self.first_run else None
                print(f"Generated CLAP score: {pred_clap}") if self.first_run else None
                self.log('gt_clap', gt_clap, on_step=True, on_epoch=True, prog_bar=True)
                self.log('pred_clap', pred_clap, on_step=True, on_epoch=True, prog_bar=True)
                
            except Exception as e:
                print(f"Error computing CLAP score: {e}")
                
                
                
                
            norm_preds = preds.mean(dim=1) / preds.mean(dim=1).norm(dim=1, keepdim=True)
            norm_latents = latents.mean(dim=1) / latents.mean(dim=1).norm(dim=1, keepdim=True)
                
            audio_to_audio_sims = norm_preds @ norm_latents.t()
            audio_to_audio_sims = audio_to_audio_sims.diag().mean()
            gt_audio_to_audio_sims = norm_latents @ norm_latents.t()
            gt_audio_to_audio_sims = gt_audio_to_audio_sims.diag().mean()
            
            self.log('A2A_CLAP', audio_to_audio_sims, on_step=True, on_epoch=True, prog_bar=True)
            self.log('GT_A2A_CLAP', gt_audio_to_audio_sims, on_step=True, on_epoch=True, prog_bar=True)
            
            self.train_audio_preds.append(preds.mean(dim=1).detach().cpu())
            self.train_gt_audio.append(latents.mean(dim=1).detach().cpu())
            self.train_gt_text.append(self.encode_text(prompt).get('projected_pooler_output', self.encode_text(prompt)['last_hidden_state'].mean(1)).detach().cpu())
            
            
            
            
        if self.scheduler is not None:
            self.scheduler.step()
        return loss
    
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        loss, loss_dict = self(latents.mean(1), prompt, validation_mode=True)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        for key in loss_dict:
            self.log(f'val_{key}', loss_dict[key], on_step=False, on_epoch=True, prog_bar=True)
        
        
        ## generate some samples for validation
        if self.current_epoch % self.retrieval_every_n_epochs == 0:
            preds = self.inference(prompt).permute(0,2,1)
            preds = preds.mean(dim=1)
            latents = latents.mean(dim=1)
            text_dict = self.encode_text(prompt)
            text_embedding = text_dict.get('projected_pooler_output', text_dict['last_hidden_state'].mean(1))
            
            
            self.val_audio_preds[dataloader_idx].append(preds.detach().cpu())
            self.val_gt_audio[dataloader_idx].append(latents.detach().cpu())
            self.val_gt_text[dataloader_idx].append(text_embedding.detach().cpu())
            
        
        return loss
    

    def on_validation_epoch_end(self, thresholds = [1,5,10]):
        on_validation_epoch_end_(self, thresholds)
        
    def on_train_epoch_end(self, thresholds = [1,5,10]):
        on_train_epoch_end_(self, thresholds)
        
    
    
    def configure_optimizers(self):
        return configure_optimizers_(self_=self)
    
    