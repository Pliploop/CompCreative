
from omegaconf import OmegaConf
from sagemaker_training.sagemaker_processing import launch_sagemaker_processing
import json
import os

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset from S3 URI.')
    parser.add_argument('--path', type=str, help='The S3 URI of the dataset', default='s3://maml-aimcdt/datasets/embeddings/mtg-jamendo/clap/1hz/')
    parser.add_argument('--output', type=str, help='The output path for the statistics', default='/opt/ml/processing/output/statistics')
    parser.add_argument('--n', type=str, help='The number of samples to process', default='-1')
    parser.add_argument('--save', type=bool, help='Save the statistics to the output path', default=True)
    parser.add_argument('--sagemaker_config', type=str, help='The path to the sagemaker config file', default='sagemaker_training/configs/dataset_statistics/preprocessing_config.yaml')
    parser.add_argument('--fixed_length', type=str, help='The fixed length of the motion features', default='64')
    
    
    args = parser.parse_args()
    
    
    cfg = OmegaConf.to_container(OmegaConf.load(args.sagemaker_config))
    
    cfg['processing_inputs'][0]['source'] = args.path
    cfg['processing_outputs'][0]['destination'] = args.path
    
    
 
    cfg['processor']['entrypoint'] += ['--save' if args.save else '', '--output', args.output,'--fixed_length', args.fixed_length]
    print(json.dumps(cfg, indent=4))
    
    
    launch_sagemaker_processing(cfg)