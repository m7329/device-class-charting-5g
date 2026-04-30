import numpy as np
import cupy as cp
import h5py
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform
import glob

from scipy import signal
import os
from tqdm import tqdm
from contextlib import contextmanager
import pickle

@contextmanager
def opened_w_error(filename, mode="r"):
    try:
        f = open(filename, mode)
    except IOError as err:
        yield None, err
    else:
        try:
            yield f, None
        finally:
            f.close()

class Load5gDataset():
    def __init__(self, feature_type='ofdm_absolutes'):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
        assert feature_type in ['ofdm_absolutes', 'obfuscation']
        self.feature_type = feature_type

    def load_channel_estimates(
        self,
        file_path,
        label_list=[],
        test_to_all_ratio=0.2,
        n_prbs=273,
        n_orus=1,
        n_rx_ant_per_oru=4,
        n_tx_ant=1,
        n_dmrs_symbols=3,
        random_subsampling=True,
        take_middle=False,
        subset_fraction=None,
        max_samples_per_label=None,
        subset_seed=1,
        feature_batch_size=100,
    ):
        test_csi_list = []
        test_label_list = []
        training_csi_list = []
        training_label_list = []
        '''
        for label_idx, label in enumerate(label_list):
            print("Load data for " + label)
            data_path = os.path.join(file_path, label)
            data_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and ".pickle" in f]

            if subset_fraction is not None or max_samples_per_label is not None:
                if subset_fraction is None:
                    subset_fraction_ = 1.0
                else:
                    subset_fraction_ = float(subset_fraction)
                    if subset_fraction_ <= 0.0 or subset_fraction_ > 1.0:
                        raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")

                n_total = len(data_files)
                n_subset = max(1, int(np.floor(n_total * subset_fraction_)))
                if max_samples_per_label is not None:
                    n_subset = min(n_subset, int(max_samples_per_label))
                n_subset = min(n_subset, n_total)

                rng = np.random.default_rng(int(subset_seed))
                subset_idx = rng.choice(n_total, size=n_subset, replace=False)
                data_files = list(np.array(data_files, dtype=object)[subset_idx])
                print(f"Subsampled {n_subset}/{n_total} pickle files for {label}")
            '''

        # (# DIFFERENCE TO EXISTING CODEBASE: ONLY USING THE SAME NUMBER OF PICKLES FOR ALL LABELS, THUS DISCARDING ALL EXCESS PICKLES)

        for label_idx, label in enumerate(label_list):
            # LIST PICKLE FOR LABEL
            pattern = os.path.join(file_path, label, '*.pickle')
            print("Pattern: ", pattern)
            paths = glob.glob(pattern)
            n_data_samples = len(paths)
            # Using float because there is no double type in Python. Float has already double precision.
            sorted_paths = sorted(paths, key=lambda p: float(os.path.basename(p).split('_')[0]))

            # DISCARD EXCESS PICKLES
            #sorted_paths = sorted_paths[:n_data_samples]
            #sorted_paths_len_list.append(len(sorted_paths))

            n_subset = n_data_samples
            if max_samples_per_label is not None:
                n_subset = min(n_subset, int(max_samples_per_label))
                rng = np.random.default_rng(int(subset_seed))
                subset_idx = rng.choice(n_data_samples, size=n_subset, replace=False)
                sorted_paths = list(np.array(sorted_paths, dtype=object)[subset_idx])
                print(f"Subsampled {n_subset}/{n_data_samples} pickle files for {label}")

            
            #H = np.zeros((n_data_samples, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols), dtype=np.complex64)
            noise_var = np.zeros((n_subset,n_orus), dtype=np.float32)
            sample_timestamps = np.zeros((n_subset), dtype=np.float64)

            print("Processing CSI estimates")

            t = tqdm(total=n_subset)
            H_list = np.zeros((n_subset, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols), dtype=np.complex64)
            # read in all pickle files and store data to associated variables
            for idx, data_file in enumerate(sorted_paths):
                t.update()  
                with opened_w_error(data_file, "rb") as (file, err):
                    if err:
                        print("File " + data_file + " has IO Error: " + str(err))
                    else:
                        x = pickle.load(file)
                        # h_sample = x['ch_est']
                        h_sample = x['ch_est'] # N_ORU x Rx ant x layer x frequency x time
                        H = np.squeeze(np.array(h_sample), axis=1)    # np.array should make a copy, slice assignment should also do the copy on itself
                        assert H.shape == (n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols), f"H shape is {H.shape} but should be ({n_orus}, {n_rx_ant_per_oru}, {n_tx_ant}, {n_prbs*12}, {n_dmrs_symbols})"
                        H_list[idx, :, :, :, :, :] = H

                        # H[idx, :, :, :, :, :] = np.take(np.squeeze(np.array(h_sample), axis=1), indices=[0,2], axis=0)    
                        # noise_var[idx, 0] = x['noise_var_dB'][0] # @TODO: here is a bug in the PyAerial Notebook! It should be the noise var from two O-RUs
                        noise_var[idx, :] = np.squeeze(np.array(x['noise_var_dB']))
                        # noise_var[idx, :] = np.take(np.squeeze(np.array(x['noise_var_dB'])), indices=[0,2], axis=0)    
                        # we also have in x the keys 'start_prb', 'num_prbs' but we use all 273 PRBs all of the times

                        timestamp_str = float(os.path.basename(data_file).split('_')[0])
                        sample_timestamps[idx] = timestamp_str
            # Ensure tqdm finishes its line before subsequent prints.
            t.close()
            print("")

            assert H.shape == (n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols), f"H shape is {H.shape} but should be ({n_orus}, {n_rx_ant_per_oru}, {n_tx_ant}, {n_prbs*12}, {n_dmrs_symbols})"
            assert H_list.shape == (n_subset, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols), f"H_list shape is {H_list.shape} but should be ({n_subset}, {n_orus}, {n_rx_ant_per_oru}, {n_tx_ant}, {n_prbs*12}, {n_dmrs_symbols})"

            n_batches = n_subset // feature_batch_size
            n_samples_per_batch = feature_batch_size
            H_list_full = H_list
            features_list = np.zeros((n_batches, n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2), dtype=np.complex64)

            for batch_idx in range(n_batches):

                #assert all(n_data_samples == sorted_paths_len_list[i] for i in range(len(sorted_paths_len_list))), "Number of data samples is not the same for all labels"
                #H_list_stacked = np.stack(H_list, axis=0) # [len(label_list)*N_PICKLES_PER_LABEL, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols]
                #assert H_list_stacked.shape[0] == len(label_list) * n_data_samples
                #H_all = np.reshape(H_list_stacked, [len(label_list), n_data_samples, *H_list_stacked.shape[1:]]) # H_all has dimension: [len(label_list), N_PICKLES_PER_LABEL, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols]

                
                # compute features
                if self.feature_type == 'ofdm_absolutes':
                    '''
                    print(f"Computing mean absolutes over N_tx={n_tx_ant} and all {n_dmrs_symbols} DMRS symbols")
                    H = np.mean(np.abs(H), axis=(3,5))

                    print("Normalize per AP")
                    H = H / np.linalg.norm(H, ord="fro", axis=(2,3), keepdims=True)

                    print("Stack APs for each CSI sample")
                    H = np.reshape(H, [n_data_samples, -1, n_prbs*12, 1])

                    features = H
                    '''
                    assert False, "ofdm_absolutes feature type is not implemented"

                elif self.feature_type == 'obfuscation':
                # H_list has dimension: [n_data_samples, n_orus, n_rx_ant_per_oru, n_tx_ant, n_prbs*12, n_dmrs_symbols]
                    H_batch = H_list_full[batch_idx * feature_batch_size:(batch_idx + 1) * feature_batch_size, :, :, :, :, :]
                    H_ = np.transpose(H_batch, [3,4,5,0,1,2])
                    assert H_.shape == (n_tx_ant, n_prbs*12, n_dmrs_symbols, n_samples_per_batch, n_orus, n_rx_ant_per_oru), f"H_ shape is {H_.shape} but should be ({n_tx_ant}, {n_prbs*12}, {n_dmrs_symbols}, {n_samples_per_batch}, {n_orus}, {n_rx_ant_per_oru})"
                    H_ = np.reshape(H_, [n_tx_ant*n_prbs*12*n_dmrs_symbols, -1])
                    # H_ is [n_tx_ant * n_prbs * 12 * n_dmrs_symbols, n_samples_per_batch * n_orus * n_rx_ant_per_oru]

                    print('Normalize each channel vector')
                    H_ = H_ / np.linalg.norm(H_, axis=-1, keepdims=True)

                    # # Gram and Eigh (for eigenvalue decomposition, computationally more intensive than SVD)
                    # R_hat = H_ @ H_.H
                    # r_eigenvalues, r_eigenvectors = np.linalg.eigh(R_hat)

                    print("Compute SVD")
                    #U, S, Vh = np.linalg.svd(H_, full_matrices=False, compute_uv=True, hermitian=False)
                    #U, S, Vh = cp.linalg.svd(cp.array(H_), full_matrices=False, compute_uv=True)
                    
                    
                    # Workaround for cuSOLVER/CuPy: computing singular vectors (compute_uv=True)
                    # can fail for complex64 in some environments. Using complex128 is robust.
                    #Hb = cp.array(H_).astype(cp.complex128)
                    U, S, Vh = np.linalg.svd(H_, full_matrices=False, compute_uv=True, hermitian=False)
                    principal_singular_vector = U[:, 0].astype(np.complex64)
                    #principal_singular_vector = cp.asnumpy(U[:, 0]).astype(np.complex64)
                    
                    assert principal_singular_vector.shape == (n_tx_ant * n_prbs * 12 * n_dmrs_symbols,), f"principal_singular_vector shape is {principal_singular_vector.shape} but should be ({n_tx_ant * n_prbs * 12 * n_dmrs_symbols,})"
                    print("SVD successful")
                    #U, S, Vh = np.linalg.svd(H_, full_matrices=False, compute_uv=True, hermitian=False)
                    #principal_singular_vector = U[:,:,0].astype(np.complex64)

                    # principal_singular_vector has dims [n_tx_ant * n_prbs * 12 * n_dmrs_symbols]
                    features = np.reshape(principal_singular_vector, [
                        n_tx_ant * n_prbs * 12, n_dmrs_symbols])
                    # features has dims [n_tx_ant * n_prbs * 12, n_dmrs_symbols]
                    assert features.shape == (n_tx_ant * n_prbs * 12, n_dmrs_symbols)

                    # I AM USING THE COMPLEX FEATURES IN MY CODE, WHILE HERE, THE RE/IM PARTS ARE STACKED #

                    print("Stack real and imaginary part in last dimension")
                    features = np.expand_dims(principal_singular_vector, axis=-1)
                    features = np.concatenate([np.real(features),np.imag(features)], axis=-1)
                    features = np.reshape(features, [n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2])
                    assert features.shape == (n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2), f"features shape is {features.shape} but should be ({n_tx_ant * n_prbs * 12}, {n_dmrs_symbols}, 2)"
                    features_list[batch_idx, :, :, :] = features

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()

            if random_subsampling:
                print("Randomly shuffle samples")
                np.random.shuffle(features_list)
            else:
                '''
                print("Sort samples according to timestamps")
                time_asc_idx = np.argsort(sample_timestamps)
                features = np.take(features, time_asc_idx, axis=0)
                '''
                pass # Already sorted by timestamp before loading .pickle files

            n_test = int(round(n_batches * test_to_all_ratio))
            if n_batches > 1 and 0.0 < test_to_all_ratio < 1.0:
                # Keep both splits non-empty for downstream training/validation code.
                n_test = max(1, min(n_test, n_batches - 1))
            n_training = n_batches - n_test

            print(f"Append CSI samples from {label} to list")
            if take_middle:
                val_idx = np.arange(n_batches//2 - int(np.floor(n_test/2)), n_batches//2 + int(np.ceil(n_test/2)))
                tr_idx = np.delete(np.arange(n_subset), val_idx)
            else:
                tr_idx = np.arange(0,n_training)
                val_idx = np.arange(n_training,n_batches)
            assert np.size(val_idx) == n_test and np.size(tr_idx) == n_training
            
            training_csi_list.append(np.take(features_list, indices=tr_idx, axis=0))
            test_csi_list.append(np.take(features_list, indices=val_idx, axis=0))
            test_label_list.append([label_idx]*n_test)
            training_label_list.append([label_idx]*n_training)

        print("Concatenate all classes")
        training_csi = np.concatenate(training_csi_list)
        test_csi = np.concatenate(test_csi_list)
        training_labels = np.concatenate(training_label_list)
        test_labels = np.concatenate(test_label_list)
        
        print("Shuffle training data set")
        training_perm_idx = np.arange(np.shape(training_csi)[0])
        np.random.shuffle(training_perm_idx)
        training_csi = np.take(training_csi, indices=training_perm_idx, axis=0)
        training_labels = np.take(training_labels, indices=training_perm_idx, axis=0)

        print("Shuffle test data set")
        test_perm_idx = np.arange(np.shape(test_csi)[0])
        np.random.shuffle(test_perm_idx)
        test_csi = np.take(test_csi, indices=test_perm_idx, axis=0)
        test_labels = np.take(test_labels, indices=test_perm_idx, axis=0)

        return [training_csi, training_labels, test_csi, test_labels]

        

