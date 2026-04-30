# THIS IS A FILE ONLY CONTAINING MULTIPLE CELLS FROM dimensionality_reduction_test.ipynb


# CELL #

# LIBRARY IMPORTS

import os
import glob
import pickle
import random

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import cupy as cp

import importlib.util
from pathlib import Path


# CELL #

# CONFIG FILE FOR IMPORTING PICKLE FILES
# HAS RANDOM SEED, EXECUTE ONCE


DATASET_BASE = '/scratch/bsc26f18/datasets/device_classification_2025_11_15'
SPLIT = 'all'  # 'training' or 'testing', or 'all'

# Choose which device folders (labels) to sample from.
LABELS = ['sgs23']

# How many pickle files to load per label (keep it small for interactive plotting).
N_PICKLES_PER_LABEL = 100

START_OFFSET = 1000

# Selection strategy: 'first' (sorted filenames) or 'random', or 'n_sorted_after_m_samples'
SELECTION_STRATEGY = 'n_sorted_after_m_samples'
my_rng = random.Random(1)

n_dmrs_symbols = 3
n_prbs = 273
n_tx_ant = 1
n_rx_ant_per_oru = 4
n_orus = 4

reshape_check = False
cupy_check = False

def list_pickles_for_label(dataset_base: str, split: str, label: str, debug: bool = False):
    """Return sorted list of pickle paths for a given label."""
    pattern = os.path.join(dataset_base, split, label, '*.pickle')
    print("Pattern: ", pattern)
    paths = glob.glob(pattern)
    # Using float because there is no double type in Python. Float has already double precision.
    sorted_paths = sorted(paths, key=lambda p: float(os.path.basename(p).split('_')[0]))
    sorted_keys = [float(os.path.basename(p).split('_')[0]) for p in sorted_paths]

    if debug:
        assert all(a<=b for a,b in zip(sorted_keys[:-1], sorted_keys[1:])), "Paths are not sorted by timestamp!"
        print("\n".join(sorted_paths[1000:1050]))
        
    return sorted_paths


def select_subset_with_indices(paths, n: int, strategy: str, rng: random.Random, m: int = 0):
    """Select paths while keeping pre-rng indices within the sorted `paths` list."""
    if n is None or n >= len(paths):
        chosen_indices = list(range(len(paths)))
    elif strategy == 'first':
        chosen_indices = list(range(n))
    elif strategy == 'n_sorted_after_m_samples':
        chosen_indices = list(range(n+m))[m:]
    elif strategy == 'random':
        chosen_indices = rng.sample(range(len(paths)), k=n)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    chosen_paths = [paths[i] for i in chosen_indices]
    return chosen_paths, chosen_indices


# CELL #

# LOAD PICKLE FILES

#samples = []
sample_paths = []
sample_rec_arr = []
sample_dicts = np.zeros((len(LABELS), N_PICKLES_PER_LABEL), dtype=object)
H_list = []

for i, label in enumerate(LABELS):

    working_dict = sample_dicts[i]

    all_paths = list_pickles_for_label(DATASET_BASE, SPLIT, label, debug=True)
    if not all_paths:
        raise FileNotFoundError(f"No pickle files found for label='{label}' under {DATASET_BASE}/{SPLIT}")
    chosen, chosen_indices = select_subset_with_indices(all_paths, N_PICKLES_PER_LABEL, SELECTION_STRATEGY, my_rng, m=START_OFFSET)
    print(f"Label={label}: found {len(all_paths)} pickles, selected {len(chosen)}")
    
    i=0
    for p, pre_rng_idx in zip(chosen, chosen_indices):
        with open(p, 'rb') as f:
            x = pickle.load(f)
    
        # Match dataset_preparation.py behavior
        h_arr = np.array(x['ch_est'])
        H = np.squeeze(h_arr, axis=1) # [n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols]
        H_list.append(H)
    
        # Useful debug info
        sample_rec_arr.append([label, pre_rng_idx, p])
        base = os.path.basename(p)
        sample_paths.append(base)

        rec = {
            'pickle_path': p,
            'label': label,
            'pre_rng_index_within_label': pre_rng_idx,
            'raw csi': h_arr,
            'H': H
        }
        working_dict[i] = rec
        i+=1


H_list_stacked = np.stack(H_list, axis=0) # [len(LABELS)*N_PICKLES_PER_LABEL, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols]
assert H_list_stacked.shape[0] == len(LABELS) * N_PICKLES_PER_LABEL

H_all = np.reshape(H_list_stacked, [len(LABELS), N_PICKLES_PER_LABEL, *H_list_stacked.shape[1:]])
# H_all has dimension: [len(LABELS), N_PICKLES_PER_LABEL, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols]


print('\nTotal loaded samples:', len(sample_rec_arr))
if len(sample_rec_arr) == 0:
    raise RuntimeError('No samples loaded')

print('Example labels:', sorted(set(s[0] for s in sample_rec_arr)))
#print('Ensure randomness:', sorted(set(s[1] for s in sample_rec_arr)))
#print('\nH_all shape:', H_all.shape)


# CELL #

# RF-FINGERPRINT FEATURE EXTRACTION

# Numpy flattens arrays in row-major order

# H_all has dimension: [len(LABELS), N_PICKLES_PER_LABEL, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols]
H_ = np.transpose(H_all, [0,1,4,5,6,2,3])
H_ = np.reshape(H_, [len(LABELS), N_PICKLES_PER_LABEL, n_tx_ant*n_prbs*12*n_dmrs_symbols, -1])
# H_ is [len(LABELS), N_PICKLES_PER_LABEL, n_tx_ant * n_prbs * 12 * n_dmrs_symbols, n_orus * n_rx_ant_per_oru]

H_ = H_ / np.linalg.norm(H_, axis=-1, keepdims=True)

U, S, Vh = np.linalg.svd(H_, full_matrices=False, compute_uv=True, hermitian=False)
principal_singular_vector = U[:,:,:,0].astype(np.complex64)


# principal_singular_vector has dims [len(LABELS), N_PICKLES_PER_LABEL, n_tx_ant * n_prbs * 12 * n_dmrs_symbols]
features = np.reshape(principal_singular_vector, [
    len(LABELS), N_PICKLES_PER_LABEL, n_tx_ant * n_prbs * 12, n_dmrs_symbols])
# features has dims [len(LABELS), N_PICKLES_PER_LABEL, n_tx_ant * n_prbs * 12, n_dmrs_symbols]

# FLATTEN FEATURES TO USE AVERAGE OVER DMRS SYMBOLS BECAUSE THE CHANGE IS ALMOST NON-EXISTENT
features_flat = np.mean(features, axis=-1, keepdims=False)
# features_flat has dims [len(LABELS), N_PICKLES_PER_LABEL, n_tx_ant * n_prbs * 12]

# NEED TO CHECK FOR CORRECTNESS OF RESHAPING
assert features_flat.shape == (len(LABELS), N_PICKLES_PER_LABEL, n_tx_ant * n_prbs * 12), "features_flat shape mismatch (numpy vs. cupy)"

global_features = np.reshape(features_flat, [-1, n_tx_ant * n_prbs * 12])
# global_features has dims [len(LABELS) * N_PICKLES_PER_LABEL, n_tx_ant * n_prbs * 12]
assert global_features.shape == (len(LABELS) * N_PICKLES_PER_LABEL, n_tx_ant * n_prbs * 12), "global_features shape mismatch"

# create local device IDs for identification later in the plot
device_ids = np.arange(len(LABELS))
global_labels = np.repeat(device_ids, N_PICKLES_PER_LABEL)
# global_labels has dims [len(LABELS) * N_PICKLES_PER_LABEL]
# because neither PCA nor Sammon's mapping shuffles or reorders the data we can use the global_labels to identify the data