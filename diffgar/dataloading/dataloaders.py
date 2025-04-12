from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from hashlib import sha256
import pandas as pd
from .datasets import TextAudioDataset
import os
from sklearn.model_selection import train_test_split
import random


def get_song_describer_annotations(data_path = None, csv_path = None, val_split = 0.1):
    
    data_path = data_path if data_path is not None else '/import/research_c4dm/jpmg86/song-describer/data/audio'
    csv_path = os.path.join(os.path.dirname(data_path), 'song_describer.csv') if csv_path is None else csv_path
    
    df = pd.read_csv(csv_path)
    
    
    df = df[['path','caption','is_valid_subset','caption_id']].rename(columns = {'path':'file_path'})
    df['file_path'] = os.path.join(data_path) + '/' + df['file_path']
    #replace .mp3 with .2min.mp3
    df['file_path'] = df['file_path'].apply(lambda x: x.replace('.mp3','.2min.mp3'))
    
    records = df.to_dict(orient = 'records')
    
    for record in records:
        record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
    if val_split == 0.0:
        print('No validation split')
        for record in records:
            record['split'] = 'train'
        return records
    
    train_indices, val_indices = train_test_split(range(len(records)), test_size = val_split, random_state = 42)
    
    for idx in train_indices:
        records[idx]['split'] = 'train'
        
    for idx in val_indices:
        records[idx]['split'] = 'val'
    
    return records



def get_musiccaps_annotations(data_path = None, csv_path = None, val_split = 0.1, test_split = 0.1):
        
        df = pd.read_csv(csv_path)
        df['file_path'] = data_path + '/' + df['ytid'] + '.wav'
        
        records = df.to_dict(orient = 'records')
        
        for record in records:
            record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
            
        if val_split == 0.0:
            print('No validation split')
            for record in records:
                record['split'] = 'train'
            return records
        
        # split into train, val, test
        train_indices, test_indices = train_test_split(range(len(records)), test_size = test_split + val_split, random_state = 42)
        val_indices, test_indices = train_test_split(test_indices, test_size = test_split/(test_split + val_split), random_state = 42)
        
        
        
        for idx in train_indices:
            records[idx]['split'] = 'train'
            
        for idx in val_indices:
            records[idx]['split'] = 'val'
            
        for idx in test_indices:
            records[idx]['split'] = 'test'
            
        return records
    
def get_musiccaps_truncated_annotations(data_path = None, csv_path = None, val_split = 0.1, test_split = 0.1):
    df = pd.read_csv(csv_path)
    df['file_path'] = data_path + '/' + df['ytid'] + '.wav'
    
    import random
    
    records = df.to_dict(orient = 'records')
    
    print('Truncating captions')
    
    for record in records:
        # select a random sentence from the caption
        sentences = record['caption'].split('.')
        random_sentence = random.choice(sentences)
        record['caption'] = {sha256(random_sentence.encode('utf-8')).hexdigest(): random_sentence}
        
    if val_split == 0.0:
        print('No validation split')
        for record in records:
            record['split'] = 'train'
        return records
    
    # split into train, val, test
    train_indices, test_indices = train_test_split(range(len(records)), test_size = test_split + val_split, random_state = 42)
    val_indices, test_indices = train_test_split(test_indices, test_size = test_split/(test_split + val_split), random_state = 42)
    
    for idx in train_indices:
        records[idx]['split'] = 'train'
        
    for idx in val_indices:
        records[idx]['split'] = 'val'
        
    for idx in test_indices:
        records[idx]['split'] = 'test'
        
    return records

def get_upmm_annotations(data_path = None, csv_path = None):
    
    data_path = data_path if data_path is not None else '/import/research_c4dm/jpmg86/upmm/data/audio'
    csv_path = os.path.join(os.path.dirname(data_path), 'upmm_captions.csv') if csv_path is None else csv_path
    
    df = pd.read_csv(csv_path)
    
    df['file_path'] = os.path.join(data_path) + '/' + df['file_path']
    
    records = df.to_dict(orient = 'records')
    
    for record in records:
        record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
    
    return records

def get_folder_annotations(data_path = None):
    
    # recursively get all files in the data_path directory that are audio files, and their paths
    audio_files = []
    
    for root, dirs, files in os.walk(data_path):
        audio_files += [os.path.join(root, file) for file in files if file.endswith('.wav') or file.endswith('.mp3')]
        
    records = [{'file_path': file, 'caption': '', 'split': 'train'} for file in audio_files]
    
    return records
    
    

class TextAudioDataModule(LightningDataModule):
    
    def __init__(self, tasks, dataloader_kwargs, return_audio = True, return_text = True, concept = None, target_n_samples = 96000, target_sr = 48000, batch_size = 32, num_workers = 0, preextracted_features = False, truncate_preextracted = 50, root_dir = None, new_dir = None):


        super().__init__()


        self.annotations = []
        for task in tasks:
            task_ = task['task']
            split_ = task.get('split', 'keep')
            annotations_ = eval(f"get_{task_}_annotations")(**task.get('task_kwargs', {}))
            for annot in annotations_:
                if split_ == 'keep':
                    continue
                else:
                    annot['split'] = split_
                    
            self.annotations.append(annotations_)
                    


        self.return_audio = dataloader_kwargs.get('return_audio', True)
        self.return_text = dataloader_kwargs.get('return_text', True)
        self.concept = dataloader_kwargs.get('concept', None)
        self.target_n_samples = dataloader_kwargs.get('target_n_samples', 96000)
        self.target_sr = dataloader_kwargs.get('target_sr', 48000)
        self.batch_size = dataloader_kwargs.get('batch_size', 32)
        self.num_workers = dataloader_kwargs.get('num_workers', 0)
        self.preextracted_features = dataloader_kwargs.get('preextracted_features', False)
        self.truncate_preextracted = dataloader_kwargs.get('truncate_preextracted', 50)
        
        
        # quick hack for a special musiccaps task
        if 'musiccaps_truncated' in [task['task'] for task in tasks]:
            truncate_preextracted = []
            for task in tasks:
                if task['task'] != 'musiccaps_truncated':
                    truncate_preextracted.append(self.truncate_preextracted) 
                else: truncate_preextracted.append(1)
                
            self.truncate_preextracted = truncate_preextracted
        else:
            self.truncate_preextracted = [self.truncate_preextracted for _ in range(len(tasks))]
        
        self.root_dirs = [task_.get('root_dir', None) for task_ in tasks]
        self.new_dirs = [task_.get('new_dir', None) for task_ in tasks]
        
        self.dataloader_names = [task_['task'] for task_ in tasks]

        # do some cleaning : we want to return a list of dictionary records.
        # each record has a file_path as a string. captions are stored as a dictionary of possible captions
        # with keys being hashes of the captions and values being the captions themselves.
        # let's start by dealing with the case where captions are strings instead, let's turn them into lists of strings

        for annotations_ in self.annotations:
            for annot in annotations_:
                if isinstance(annot['caption'], str):
                    annot['caption'] = [annot['caption']]

            for annot in annotations_:
                if isinstance(annot['caption'], list):
                    annot['caption'] = {sha256(caption.encode('utf-8')).hexdigest(): caption for caption in annot['caption']}
        
        self.train_annotations = []
        self.val_annotations = []
        self.test_annotations = []
        
        
        for dataset_ in self.annotations:
            self.train_annotations.append([annot for annot in dataset_ if annot['split'] == 'train'])
            self.val_annotations.append([annot for annot in dataset_ if annot['split'] == 'val'])
            self.test_annotations.append([annot for annot in dataset_ if annot['split'] == 'val'])
        
        print(f"Number of training samples: {sum([len(annot) for annot in self.train_annotations])} over {len(self.train_annotations)} datasets, {[len(annot) for annot in self.train_annotations]}")
        print(f"Number of validation samples: {sum([len(annot) for annot in self.val_annotations])} over {len(self.val_annotations)} datasets, {[len(annot) for annot in self.val_annotations]}")
        print(f"Number of test samples: {sum([len(annot) for annot in self.test_annotations])} over {len(self.test_annotations)} datasets, {[len(annot) for annot in self.test_annotations]}")
        print(f"Datasets: {self.dataloader_names}")
        
        
    def setup(self, stage: str) -> None:
        # self.train_dataset = TextAudioDataset(annotations=self.train_annotations, target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted, root_dir=self.root_dir, new_dir=self.new_dir)
        # self.val_dataset = TextAudioDataset(annotations=self.val_annotations, target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted, root_dir=self.root_dir, new_dir=self.new_dir)
        # self.test_dataset = TextAudioDataset(annotations=self.test_annotations, target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted, root_dir=self.root_dir, new_dir=self.new_dir)
        
        self.train_dataset = TextAudioDataset(annotations=self.train_annotations[0], target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted[0], root_dir=self.root_dirs[0], new_dir=self.new_dirs[0])
        self.val_datasets = [TextAudioDataset(annotations=self.val_annotations[i], target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted[i], root_dir=self.root_dirs[i], new_dir=self.new_dirs[i]) for i in range(len(self.val_annotations))]
        self.test_datasets = [TextAudioDataset(annotations=self.test_annotations[i], target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted[i], root_dir=self.root_dirs[i], new_dir=self.new_dirs[i]) for i in range(len(self.test_annotations))]
        
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return [DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False) for val_dataset in self.val_datasets]
    
    def test_dataloader(self):
        return [DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False) for test_dataset in self.test_datasets]