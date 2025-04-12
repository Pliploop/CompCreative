import boto3
import wandb
from diffgar.dataloading.dataloaders import TextAudioDataModule
from diffgar.models.ldm.diffusion import LightningDiffGar,LightningMLPldm,LightningMLPMSE,LightningMSEGar
from rich.pretty import pprint

def load_model_and_dataset_eval(model_name, model_step, task, device='cuda:4', return_stats_path = False, return_full_dataset = False):

    model_step = model_step if model_step is not None else 100000

    path = f's3://maml-aimcdt/storage/julien/DiffGAR/training_checkpoints/{model_name}'
    experiment_name = path.split('/')[-1]
    config_path = path + '/config.yaml'
    
    # get the keys in the bucket
    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket='maml-aimcdt', Prefix=f'storage/julien/DiffGAR/training_checkpoints/{model_name}')
    keys = [obj['Key'] for obj in objects.get('Contents', [])]
    
    
    #get the key where 'best' is in the name
    p = [key for key in keys if 'best' in key]
    p = p[0] if len(p) > 0 else None
    object_name = p.split('/')[-1] if p is not None else ''
    
    
    ckpt_path = path + f'/checkpoint-step={model_step}-recent.ckpt' if model_step != 'best' else path + f'/{object_name}'
    print(ckpt_path)

    # get the config from wandb
    api = wandb.Api()
    
    
    runs = api.runs(f'jul-guinot/DiffGAR-LDM')
    # get the config where names match the experiment name
    for run in runs:
        if run.name == experiment_name:
            config = run.config
            
    model_cls = eval(config['model'].get('class_path', 'LightningDiffGar').split('.')[-1])
    pprint(model_cls)

    if 'init_args' in config['model']:
        config['model'] = config['model']['init_args']
    
    # pprint(config['model'])
    channels = config['model']['unet_model_config']['in_channels'] if 'unet_model_config' in config['model'] else config['model']['mlp_model_config']['in_channels']

    training_encoder_pair = config['model']['encoder_pair']

    encoder_pair_to_new_dir = {
        'song_describer': {
            'muleT5':   '/import/research_c4dm/jpmg86/song-describer/data/muleproj/npy/1hz',
            'clap':    '/import/research_c4dm/jpmg86/song-describer/data/clap/npy',
            'music2latent' : f'/import/research_c4dm/jpmg86/embedding_datasets/song-describer/data/music2latent/{channels}/1hz',
            'MusCALL': '/import/research_c4dm/jpmg86/song-describer/data/muscall/npy/1hz'
        },
        'musiccaps': {
            'muleT5':   '/import/research_c4dm/jpmg86/musiccaps/mule_npy/1hz',
            'clap':    '/import/research_c4dm/jpmg86/musiccaps/clap_npy/1hz',
            'MusCALL': '/import/research_c4dm/jpmg86/musiccaps/muscall/1hz'
        }
    }

    encoder_pair_to_old_dir = {
        'song_describer': '/import/research_c4dm/jpmg86/song-describer/data/audio',
        'musiccaps': '/import/c4dm-datasets/musiccaps/musiccaps_10s',
    }
    
    task_to_task_kws = {
        'musiccaps': {
                'data_path': '/import/c4dm-datasets/musiccaps/musiccaps_10s',
                'csv_path': '/import/c4dm-datasets/musiccaps/musiccaps-public.csv'
        },
        'song_describer': {
                'data_path': '/import/research_c4dm/jpmg86/song-describer/data/audio',
                'csv_path': '/import/research_c4dm/jpmg86/song-describer/data/song_describer.csv'
        }
    }
    
    encoder_pair_to_stats_path = {
        'clap': 's3://maml-aimcdt/datasets/embeddings/spotify_most_popular/clap/1hz/statistics',
        'muleT5': 's3://maml-aimcdt/datasets/embeddings/spotify_most_popular/mule_512_norm/1hz/statistics'
    }
        

    
    model = model_cls.from_pretrained(config_path, ckpt_path, device=device)
    model.to(device)
    
    
    config = {
        'tasks' : [
            {
                'task': task,
                'task_kwargs': task_to_task_kws[task],
                'split': 'test' if return_full_dataset else 'keep',
                'root_dir': encoder_pair_to_old_dir[task],
                'new_dir': encoder_pair_to_new_dir[task][training_encoder_pair],
            }
        ],
        'dataloader_kwargs': {
            'batch_size': 1,
            'preextracted_features': True,
            'truncate_preextracted': 64
        }
    }

    # latent_dm = TextAudioDataModule(
    #     task=task,
    #     task_kwargs=task_to_task_kws[task],
    #     batch_size=1,
    #     preextracted_features=True,
    #     truncate_preextracted=64,
    #     new_dir=encoder_pair_to_new_dir[task][training_encoder_pair],
    #     root_dir=encoder_pair_to_old_dir[task]
    # )
    
    latent_dm = TextAudioDataModule(**config)
    
    # if return_full_dataset:
    #     latent_dm.test_annotations = latent_dm.annotations
    #     latent_dm.val_annotations = latent_dm.annotations
    
    latent_dm.setup(None)

    if task == 'song_describer':
        dataset = latent_dm.val_datasets[0]
    else:
        # might need to modify musiccaps so that it is all in the test set
        dataset = latent_dm.test_datasets[0]
        
    
        
    

    if return_stats_path:
        return model, dataset, experiment_name, config, encoder_pair_to_stats_path[training_encoder_pair]
    return model, dataset, experiment_name, config


