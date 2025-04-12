## here we will open a json file, read it, and log the results to weights and biases
import json
import os
import wandb
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file-postfix", type=str, default="") # this is the file postfix to be used to find the file to log
parser.add_argument("--task", type=str, default="song_describer") # this is the file postfix to be used to find the file to log
args = parser.parse_args()

project = "DiffGAR-LDM-eval"
skip = True
task  = args.task

file_path = f"results/retrieval/{task}/metrics{args.file_postfix}.json"
file_name = os.path.basename(file_path)
   

try:
    with open(file_path, "r") as f:
        results = json.load(f)
except:
    results = {}

results = results.get(task, {})
runs = wandb.Api().runs(f"{project}")
print(f"Found {len(runs)} runs")
runs_and_configs = []
for run in tqdm(runs):
    runs_and_configs.append ({'config': run.config, 'steps': run.history(keys=None)['training_step'].tolist(),'name': run.name, 'id': run.id})

def reformat_dict(data):
    keys = next(iter(data.values())).keys()  # Get the inner keys (e.g., '1', '3')
    new_structure = {}
    for key in keys:
        new_structure[key] = {'k': key}
        for outer_key, outer_value in data.items():
            new_structure[key][outer_key] = outer_value.get(key, {})
    return list(new_structure.values())


print(f'Found {len(results)} results for {task}')

for key, runs_for_model in results.items():
    
    model_name = '-'.join(key.split("-")[:-2])
    steps = int(key.split("-")[-2])
    
    
    
    for run in runs_for_model:
        
        metrics = run['metrics']
        config = {k:v for k,v in run.items() if k != 'metrics'}
        config['task'] = task
        config['file_name'] = file_name
        
        old_key_to_new = {
            'diagonals': 'positives',
            'averages': 'negatives',
            'retrieve_gt_audio_from_gt_text': 'T2A(ground_truth)',
            'retrieve_gt_text_from_gt_audio': 'A2T(ground_truth)',
            'retrieve_gt_text_from_pred_audio': 'A2T(pred)',
            'retrieve_gt_audio_from_pred_text': 'T2A(pred)',
            
        }
        
        for old_key, new_key in old_key_to_new.items():
            if old_key in metrics:
                metrics[new_key] = metrics.pop(old_key)
                
        #create a new wandb run with the config if it doesn't exist, else update the existing run
        
        metrics = reformat_dict(metrics)
        
        # log metrics for the init run at the given step
        for metric in metrics:
            
            
            metric_config = {k:v for k,v in config.items()}
            metric_config['k'] = metric.pop('k')
            metric_config.pop('training_steps')
            for key, value in metric_config.items():
                try:
                    metric[key] = float(metric[key])
                except:
                    pass
            
            # if the run exists, get the id and update the run
            
            # print(json.dumps(metric, indent=4))
            
            run_exists = False
            
            
            
            for run__ in runs_and_configs:
                
                hist_steps = run__['steps']
                id_ = run__['id']
                config_ = run__['config']
                
                # print(model_name,json.dumps(metric_config, indent=4)) if ('1y2' in model_name and metric_config['num_samples_per_prompt'] == 100) else None
                # print(json.dumps(config_, indent=4)) if ('1y2' in model_name and config_['num_samples_per_prompt']) == 100 else None
                
                if config_ == metric_config:
                    # print('here')
                    if steps not in hist_steps:
                        print(f"Updating {model_name} at step {steps}")
                        wandbrun = wandb.init(project=project, id=id_, resume="must", config=metric_config)
                        wandb.define_metric("training_step")
                        metric['training_step'] = steps
                        wandbrun.log(metric)
                        wandbrun.finish()
                        
                        # add the run to runs_and_configs
                        for run in runs_and_configs:
                            if run['id'] == id_:
                                run['steps'].append(steps)
                        
                        pass
                    else:
                        print(f"Skipping {model_name} at step {steps} as it already exists")
                    run_exists = True
                
            if not run_exists:
                print(f"Creating {model_name} at step {steps}")
                wandbrun = wandb.init(project=project, name=model_name, config=metric_config)
                wandb.define_metric("training_step")
                metric['training_step'] = steps
                # set the step metric to be the training step for all metrics
                for m_ in metric:
                    wandb.define_metric(m_, step_metric="training_step")
                
                wandbrun.log(metric)
                runs_and_configs.append({'config': metric_config, 'steps': [steps], 'name': model_name, 'id': wandbrun.id})
                
                wandbrun.finish()
                pass
