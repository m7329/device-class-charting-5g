import numpy as np

def dim_red_pca(Features_dmrs_averaged):
    """
    NumPy-only PCA embedding for complex CSI features.

    Input:
        Features: real-valued tensor with shape
            [n_labels * n_data_samples, n_tx_ant * n_prbs * 12],
            we assume that the features are already averaged over the DMRS symbols.

    Output:
        2D (complex) map for all samples with shape:
            [D_prime=2, n_data_samples].
    """

    Features = np.asarray(Features_dmrs_averaged)
    if Features.ndim != 2:
        raise ValueError(
            "Features must have shape [n_labels * n_data_samples, n_tx_ant*n_prbs*12]. "
            f"Got shape {Features.shape}."
        )

    d_prime = 2  # Hardcoded embedding dimension.

    # Build complex features: [n_samples, n_features, n_dmrs_symbols]
    # complex_features = Features[..., 0] + 1j * Features[..., 1]
    
    n_samples = Features_dmrs_averaged.shape[0]


    # Average over DMRS symbols (as in Studer's paper with raw 2nd moment)
    #edited_features = complex_features.mean(axis=2) # [n_samples, n_features]

    # Flatten per-sample features (same as tf.reshape(complex_features, [n_samples, -1]))
    # edited_features = complex_features.reshape(n_samples, -1)  # [n_samples, n_tx_ant * n_prbs * 12 * n_dmrs_symbols]

    F = Features_dmrs_averaged.T # [n_features, n_samples]
    row_mean = F.mean(axis=1, keepdims=True)
    F_bar = F - row_mean
    F_bar_H = F_bar.conj().T # [n_samples, n_features]
    cov_matrix = F_bar @ F_bar_H # [n_features, n_features] (use this approach because n_features < n_samples)

    # Use eigh instead of svd because Hermitian matrix and more efficient
    eigenvalues, W = np.linalg.eigh(cov_matrix)
    idx_desc = np.argsort(eigenvalues)[::-1]
    #eigenvalues = eigenvalues[idx_desc]
    W = W[:, idx_desc]

    #sigma_d = eigenvalues[:d_prime]
    W_d = W[:, :d_prime]  # [n_features, 2]

    #sigma_d_complex = np.sqrt(sigma_d).astype(np.complex128)  # [2]
    #Z_pca = (sigma_d_complex * U_d).conj().T  # [2, n_features]

    Z_pca = W_d.conj().T @ F_bar # [2, n_samples]

    # Convert complex PCA scores to a real 2D map: [n_samples, D', 2].
    # Z_pca_out = np.stack([Z_pca.real, Z_pca.imag], axis=-1).astype(np.float32)

    return Z_pca