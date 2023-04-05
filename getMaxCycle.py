import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import torch
import os

from packages.physionet_importer import getTrainingSet

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




df = getTrainingSet('PhysioNet')
max_duration = 0
min_duration = 9999
for row in df.itertuples():
    aud = torchaudio.load(row.relative_path)
    states_df = pd.read_csv(row.relative_path.replace('.wav', '.csv'), header=None)
    states = states_df[0].to_numpy()

    aud = resample(aud, 1000)
    cycles = getCycles(states)
    for i, c in enumerate(cycles):
        if(c[1] - c[0] > max_duration):
            max_duration = c[1] - c[0]
            print('max: ' + str(max_duration))
        if(c[1] - c[0] < min_duration):
            min_duration = c[1] - c[0]
            print('min: ' + str(min_duration))

# min: 323
# max: 2479