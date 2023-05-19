import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import torch
import os

from packages.pascal_importer import getData

def resample(aud, newsr):

        #Resample the audio to the newsr frequency

        sig, sr = aud
        
        if(sr == newsr):
            return((sig, sr))

        num_channels = sig.shape[0]
        resig_fn = torchaudio.transforms.Resample(sr, newsr)
        resig = resig_fn(sig[:1, :])
        if(num_channels > 1):
            retwo_fn = torchaudio.transforms.Resample(sr, newsr)
            retwo = retwo_fn(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return((resig, newsr))

def getCycles(states):
    cycles = []
    debounce = False
    previous = None
    sig_len = states.shape[0]
    for i in range(sig_len):
        if (states[i] == 1 and debounce == False):
            if(previous != None):
                cycles.append([previous, i-1])
            previous = i
            debounce = True
        elif (states[i] == 4 and i == sig_len-1):
            if(previous != None):
                cycles.append([previous, i])
        elif (states[i] != 1 and debounce == True):
            debounce = False
    return cycles

def extractCyles(aud, cycles):
    sig, sr = aud
    c_signals = []
    for i, c in enumerate(cycles):
        c_sig = sig[0][c[0]:c[1]]
        c_signals.append(c_sig.unsqueeze(0))
    return(c_signals)



df = getData('PASCAL')
reference_df = pd.DataFrame()

os.makedirs('PASCALCycles', exist_ok=True)
j = 0
for row in df.itertuples():
    aud = torchaudio.load(row.relative_path)
    states_df = pd.read_csv(row.relative_path.replace('.wav', '.csv'), header=None)
    states = states_df[0].to_numpy()

    aud = resample(aud, 1000)
    cycles = getCycles(states)
    c_signals = extractCyles(aud, cycles)
    j += 1
    print(j)
    for i, c_sig in enumerate(c_signals):
        file_name = row.file_name.replace('.wav', '_' + str(i) + '.wav')
        savepath = 'PASCALCycles/' + file_name
        new_row_df = pd.DataFrame({'file_name': [file_name], 'classID': [row.classID], 'relative_path': [savepath]})
        reference_df = pd.concat([reference_df, new_row_df], ignore_index=True)
        torchaudio.save(savepath, c_sig, 1000)

reference_df.to_csv('PASCALCycles/REFERENCE.csv')