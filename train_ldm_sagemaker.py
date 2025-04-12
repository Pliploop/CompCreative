
from diffgar.dataloading.dataloaders import TextAudioDataModule
from pytorch_lightning.cli import SaveConfigCallback, LightningCLI
import yaml
import os
from jsonargparse import lazy_instance
from pytorch_lightning.strategies import DDPStrategy
import wandb
import boto3
from botocore.exceptions import NoCredentialsError
from rich.pretty import pprint

import logging

from sagemaker_training.sagemaker_training import launch_sagemaker_training
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

import json

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self) -> None:
        
            config = self.parser.dump(self.config, skip_none=False)
            print(json.dumps(config, indent=4))
            
            # with open(self.config_filename, "w") as config_file:
            #     config_file.write(config)
                
            
            
            
            

class MyLightningCLI(LightningCLI):
    
    trainer_defaults = {
        "strategy": lazy_instance(DDPStrategy, find_unused_parameters=False),
    }
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--log", default=False)
        parser.add_argument("--log_model", default=False)
        parser.add_argument("--ckpt_path", default="checkpoints")
        parser.add_argument("--resume_id", default=None)
        parser.add_argument("--resume_from_checkpoint", default=None)
        parser.add_argument("--project", default="DiffGAR-LDM")
        parser.add_argument("--test", default=False)
        parser.add_argument("--callbacks", default=[])

    def instantiate_classes(self) -> None:
        pass

if __name__ == "__main__":
    
    from diffgar.models.ldm.diffusion import LightningDiffGar, LightningMSEGar, LightningMLPldm, LightningMLPMSE

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    
    cli = MyLightningCLI(datamodule_class=TextAudioDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},trainer_defaults=MyLightningCLI.trainer_defaults)
    
    cli.parser.save(cli.config, "config.yaml", skip_none=False, overwrite=True)
    
    cfg = OmegaConf.to_container(OmegaConf.load("sagemaker_training/configs/config.yaml"))
    
    
    upload_cfg_to = cfg['pull_config']
    
    config_ = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    
    pprint(config_)
    
    s3_client = boto3.client('s3')
    # copy the config file to the s3 bucket
    try:
        #upload cfg to is of the form s3://bucket/key
        bucket, key = upload_cfg_to.replace("s3://", "").split("/", 1)
        s3_client.upload_file("config.yaml", bucket,key)
        # remove the local config file
        os.remove("config.yaml")
    except NoCredentialsError:
        print("No AWS credentials found. Please set up your AWS credentials.")
    
    launch_sagemaker_training(cfg)
    
    