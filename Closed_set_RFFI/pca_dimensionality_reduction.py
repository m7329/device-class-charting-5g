import numpy as np

def dim_red_pca(Features):
    """
    NumPy-only PCA embedding for complex CSI features.

    Input:
        Features: real-valued tensor with shape
            [n_data_samples, n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2]
        where the last dimension is [real, imag].

    Output:
        2D map for all samples with shape:
            [n_data_samples, D_prime=2, 2]
        where the last dimension is [real, imag] of the PCA scores.
    """
    # Support TensorFlow EagerTensors without importing tensorflow.
    if hasattr(Features, "numpy"):
        Features = Features.numpy()

    Features = np.asarray(Features)
    if Features.ndim != 4 or Features.shape[-1] != 2:
        raise ValueError(
            "Features must have shape [n_data_samples, n_tx_ant*n_prbs*12, n_dmrs_symbols, 2]. "
            f"Got shape {Features.shape}."
        )

    d_prime = 2  # Hardcoded embedding dimension.

    # Build complex features: [n_samples, n_features, n_dmrs_symbols]
    complex_features = Features[..., 0] + 1j * Features[..., 1]
    n_samples = complex_features.shape[0]


    # Average over DMRS symbols (as in Studer's paper with raw 2nd moment)
    edited_features = complex_features.mean(axis=2) # [n_samples, n_features]

    # Flatten per-sample features (same as tf.reshape(complex_features, [n_samples, -1]))
    # edited_features = complex_features.reshape(n_samples, -1)  # [n_samples, n_tx_ant * n_prbs * 12 * n_dmrs_symbols]

    # Assume n_x to be either n_features or n_tx_ant * n_prbs * 12 * n_dmrs_symbols,
    # depending on how to handle the averaging/flattening above

    F = edited_features.T # [n_x, n_samples]
    row_mean = F.mean(axis=1, keepdims=True)
    F_bar = F - row_mean
    F_bar_H = F_bar.conj().T # [n_samples, n_x]
    gram_matrix = F_bar_H @ F_bar # [n_samples, n_samples]

    # Use eigh instead of svd because Hermitian matrix and more efficient
    eigenvalues, U = np.linalg.eigh(gram_matrix)
    idx_desc = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_desc]
    U = U[:, idx_desc]

    sigma_d = eigenvalues[:d_prime]
    U_d = U[:, :d_prime]  # [n_samples, 2]

    sigma_d_complex = np.sqrt(sigma_d).astype(np.complex128)  # [2]
    Z_pca = (sigma_d_complex * U_d).conj().T  # [2, n_samples]

    # Convert complex PCA scores to a real 2D map: [n_samples, D', 2].
    Z_pca_out = np.stack([Z_pca.real, Z_pca.imag], axis=-1).astype(np.float32)
    return Z_pca_out