#!/usr/bin/env python
# coding: utf-8

import music2latent
import librosa
import soundfile as sf
import torchaudio
from torchaudio.functional import resample

def process_file_path(m2l_model, file_path):
    
    # Load the audio file using soundfile and resample with torchaudio
    wv, sr = torchaudio.load(file_path)
    # Resample the audio to 44.1kHz if necessary
    if sr != 44100:
        wv = resample(wv, orig_freq=sr, new_freq=44100)
        sr = 44100
        
    
    # Encode the audio to latent representation
    latent = m2l_model.encode(wv)
    
    return latent.permute(0, 2, 1) # (batch_size, sequence_length, dim)

def process_folder(folder_path, save_path=None, file_exts=['.wav', '.mp3'], dry = False, device='cpu', verbose = False):
    """
    recursively process all .wav and .mp3 files in a folder
    if save path is not None, save the latents with the same structure as the original folder to npy
    """
    import os 
    import numpy as np
    from tqdm.rich import tqdm
    from pathlib import Path
    from music2latent import EncoderDecoder
    
    # Create the encoder-decoder model
    encdec = EncoderDecoder(device = device)
    # Get the list of all files in the folder
    files = []
    for ext in file_exts:
        files.extend(Path(folder_path).rglob(f'*{ext}'))
        
    #shuffle
    files = np.random.permutation(files)
        
    # Create the save path if it doesn't exist
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    # Process each file
    for file_path in tqdm(files):
        # Process the file to get the latent representation
        
        # Save the latent representation if save_path is provided
        if save_path is not None:
            # Create the corresponding save path
            relative_path = os.path.relpath(file_path, folder_path)
            
            # print(relative_path)
            save_file_path = os.path.join(save_path, relative_path)
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            save_path_ = save_file_path.replace(file_exts[0], '.npy')
            # print(f"Processing {file_path} to {save_path}")
            
            if not os.path.exists(save_path_):
                latent = process_file_path(encdec, file_path) if not dry else None
                np.save(save_path_, latent.cpu().numpy()) if not dry else None
            else:
                print(f"File {save_path_} already exists, skipping.") if verbose else None
                latent = None
            if dry:
                print(f"Would save to {save_path_} with shape {latent.shape}") if latent is not None else None
                
    return

            
    
    
    


# In[13]:

#set 1 thread with OPM
if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    folder_path = '/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3'
    save_path = '/import/research_c4dm/jpmg86/CompCreative/mtg-jamendo-music2latent_test'

    process_folder(folder_path, save_path, file_exts=['.mp3'], dry = False, device='cuda:2', verbose = True)


# In[11]:


# In[ ]:




