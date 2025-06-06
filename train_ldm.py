from diffgar.models.ldm.diffusion import LightningLDM
from diffgar.dataloading.dataloaders import TextAudioDataModule
from pytorch_lightning.cli import SaveConfigCallback, LightningCLI
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import os
from jsonargparse import lazy_instance
from pytorch_lightning.strategies import DDPStrategy
import wandb
import boto3
from botocore.exceptions import NoCredentialsError

def instanciate_callbacks(callbacks = [], dir_path = None):
    instanciated = []
    for callback in callbacks:
        class__ = callback['__class__']
        del callback['__class__']
        if class__ == 'ModelCheckpoint':
            instanciated.append(ModelCheckpoint(dir_path, **callback))
        else:
            instanciated.append(eval(class__)(**callback))
            
    print('-------------------')
    print(instanciated)
    print('-------------------')
    return instanciated
        


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.logger is not None:
            experiment_name = trainer.logger.experiment.name
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            with open(self.config_filename, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                if trainer.global_rank == 0:
                    trainer.logger.experiment.config.update(config, allow_val_change=True)
            
            # Check if the target config path is an S3 bucket
            if self.config['ckpt_path'].startswith('s3://'):
                
    
                bucket, key = self.config['ckpt_path'].replace("s3://", "").split("/", 1)
                key = f"{key}/{experiment_name}/config.yaml"
                
                print(f"Uploading config to s3://{bucket}/{key}")
                
                s3_client = boto3.client('s3')
                try:
                    s3_client.upload_file(self.config_filename, bucket, key)
                except NoCredentialsError:
                    print("No AWS credentials found. Please set up your AWS credentials.")
            else:
                # Save the config locally
                with open(os.path.join(os.path.join(self.config['ckpt_path'], experiment_name), "config.yaml"), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)
                
            # remove local config file
            os.remove(self.config_filename)
            
            

class MyLightningCLI(LightningCLI):
    
    
    from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
    
    trainer_defaults = {
        "strategy": lazy_instance(DDPStrategy, find_unused_parameters=False),
        "callbacks": [RichProgressBar(), RichModelSummary(max_depth=2)],
    }
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--log", default=False)
        parser.add_argument("--log_model", default=False)
        parser.add_argument("--ckpt_path", default="checkpoints")
        parser.add_argument("--resume_id", default=None)
        parser.add_argument("--resume_from_checkpoint", default=None)
        parser.add_argument("--project", default="CreativeComp")
        parser.add_argument("--test", default=False)
        parser.add_argument("--callbacks", default=[])


if __name__ == "__main__":
    
    #get the model class
    
    
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    cli = MyLightningCLI(datamodule_class=TextAudioDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},trainer_defaults=MyLightningCLI.trainer_defaults)



    if cli.config.log:
        logger = WandbLogger(project=cli.config.project, id=cli.config.resume_id)
        if cli.trainer.global_rank == 0:
            experiment_name = logger.experiment.name
        else:
            api = wandb.Api()
            runs = api.runs(cli.config.project)
            latest_run = runs[0]
            experiment_name = latest_run.name
        ckpt_path = cli.config.ckpt_path
    else:
        logger = None
        ckpt_path = cli.config.ckpt_path
        experiment_name = "no_wandb"


    cli.trainer.logger = logger
    
    
    
    print("checkpoint path",ckpt_path)
    print("experiment name",experiment_name)    
    # recent_callback_step_latest = ModelCheckpoint(
    #             dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
    #             filename='checkpoint-{step}-recent',  # This means all checkpoints are saved, not just the top k
    #             every_n_train_steps = 5000,  # Replace with your desired value
    #             save_top_k = -1
    #         )
    
    #validation loss callback
    # best_callback = ModelCheckpoint(
    #     dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
    #     filename='checkpoint-{step}-best',  # This means all checkpoints are saved, not just the top k
    #     monitor='val_loss',
    #     mode='min',
    #     every_n_train_steps=2500,
    #     save_top_k=1
    # )
            
    # recent_callback_step = ModelCheckpoint(
    #     dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
    #     filename='checkpoint-{step}',  # This means all checkpoints are saved, not just the top k
    #     every_n_train_steps = 200000,  # Replace with your desired value
    #     save_top_k = 1
    # )
    
    #early stopping callback on 10000 steps
    
    # patience = cli.config.patience if hasattr(cli.config, 'patience') else 10
    
    # early_stopping_callback = EarlyStopping(
    #     monitor='val_loss',
    #     patience=patience,
    #     verbose=True,
    #     mode='min'
    # )
    
    
    ##rich progress bar callback
    from pytorch_lightning.callbacks import RichProgressBar
    
    
    
    callbacks = []
    if cli.config.log_model:
        callbacks += instanciate_callbacks(cli.config.callbacks, os.path.join(ckpt_path, experiment_name))
        
    cli.trainer.callbacks = cli.trainer.callbacks[:-1]
    
    if cli.config.log:
        cli.trainer.callbacks = cli.trainer.callbacks+callbacks
        
    

    try:
        if not os.path.exists(os.path.join(ckpt_path, experiment_name)) and 's3:' not in ckpt_path:
            os.makedirs(os.path.join(ckpt_path, experiment_name))
    except:
        pass
    
    #datamodule setup
    cli.datamodule.setup(None)
    
    
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.resume_from_checkpoint)
    
    if cli.config.test:
        cli.trainer.test()
