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
import music2latent



from diffgar.models.clap.src.laion_clap import CLAP_Module


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

def on_validation_epoch_end_(self_, thresholds = [1,5,10]):
    
    
    #only every 50 epochs
    if self_.current_epoch % self_.generate_every_n_epochs != 0:
        return
    
    pass
    
def on_train_epoch_end_(self_, thresholds = [1,5,10]):
    
    if self_.current_epoch % self_.generate_every_n_epochs != 0:
        return
    
    pass
    
def get_encoder_pair(encoder_pair, encoder_pair_kwargs=None):
    
    if encoder_pair == "clap":
        return CLAP_Module(**encoder_pair_kwargs)
    
def get_text_encoder(text_encoder, text_encoder_kwargs=None):
    if text_encoder == 'T5':
        return T5TextEncoder(**text_encoder_kwargs)
    

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
        

        if self.freeze_encoders:
            with torch.no_grad():
                encoded_text_dict = self.text_encoder.get_text_embedding(prompt, use_tensor = True, return_dict = True) if self.text_encoder is not None else self.encoder_pair.get_text_embedding(prompt, use_tensor = True, return_dict = True)
        else:
            encoded_text_dict = self.text_encoder.get_text_embedding(prompt, use_tensor = True, return_dict = True) if self.text_encoder is not None else self.encoder_pair.get_text_embedding(prompt, use_tensor = True, return_dict = True)
            
        if subject_masks is not None:
            encoded_text_dict['subject_masks'] = subject_masks

        return encoded_text_dict
            
class LDM(BaseModule):
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
        latent_length=512,
        tag_conditioning=True, # if true, splits the text embedding before encoding
        device=None,
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
        self.latent_length = latent_length
        self.tag_conditioning = tag_conditioning
        
        self.ae = music2latent.EncoderDecoder()
        
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
        
        if device is not None:
            self.to(device)   
            
            
    def ae_encode(self, audio):
        return self.ae.encode(audio).mean(0).permute(1,0)
    
    
    def ae_decode(self, latents):
        return self.ae.decode(latents.permute(1,0))     
           

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
            
            
        loss_dict = {
            'mse_loss': mse_loss,
        }
        loss = mse_loss
        

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

        # all_latents = torch.stack(all_latents, dim=0)
        # if return_all_latents:
        #     return latents, all_latents
        # else:
        #     return latents
        
        return self.ae_decode(latents)
    
    @torch.no_grad()
    def invert(
        self,
        audio,
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
    
        latents = self.ae_encode(audio)
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
        shape = (batch_size, num_channels_latents, self.latent_length)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents
    
    
class LightningLDM(LDM,LightningModule):
    
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
                generate_every_n_epochs = 50,
                tag_conditioning=True,
                latent_length=512,
                **kwargs
                ):
        
        LDM.__init__(self,
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
                    contrastive_loss_kwargs=contrastive_loss_kwargs,
                    latent_length=latent_length,
                    tag_conditioning=tag_conditioning,)
        
        self.preextracted_latents = preextracted_latents
        self.gen_examples = True
        self.first_run = True
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.generate_every_n_epochs = generate_every_n_epochs
        
        self.train_preds = []
        self.val_preds = []
        
        
    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        
        
        
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
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
            
        # if self.current_epoch % self.generate_every_n_epochs == 0 and self.gen_examples and len(self.train_audio_preds) <  self.trainer.limit_val_batches:
        #     preds = self.inference(prompt, self.inference_scheduler, num_steps = 50, disable_progress = False, guidance_scale = self.unet_model_config['infer_classifier_free_guidance_strength'])
        #     preds = preds.permute(0,2,1)
            
        if self.scheduler is not None:
            self.scheduler.step()
        return loss
        
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        
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
        # if self.current_epoch % self.generate_every_n_epochs:
        #     preds = self.inference(prompt, self.inference_scheduler, num_steps = 50, disable_progress = False, guidance_scale = self.unet_model_config['infer_classifier_free_guidance_strength'])
        #     preds = preds.permute(0,2,1)
        #     # decode with music2latent
            
        
        return loss
    
        
    def configure_optimizers(self):
        return configure_optimizers_(self_=self)
    
    
    def on_validation_epoch_end(self, thresholds = [1,5,10]):
        on_validation_epoch_end_(self, thresholds)
        
    def on_train_epoch_end(self, thresholds = [1,5,10]):
        on_train_epoch_end_(self, thresholds)
        
    
    