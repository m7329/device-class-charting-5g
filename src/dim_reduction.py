import tensorflow as tf

# TODO:
# - Test functionality of existing code
# - Test PCA function
# - Find out what Feature matrices to use for Frobenius norm calculation

# The Features tensor has dimension [n_data_samples, n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2]
# Second argument needs to be a Python integer, not a Tensor!
@tf.function
def dim_red_pca(Features, D_prime=2):
    complex_Features = tf.complex(Features[:, :, :, 0], Features[:, :, :, 1])

    n_samples = tf.shape(Features)[0]

    Z_pca_array = tf.TensorArray(tf.complex64, size=n_samples)

    # Need to iterate over first dimension of complex_Features and apply the PCA to each data sample
    for i in tf.range(n_samples):

        # We treat n_tx_ant * n_prbs * 12 as the features. Thus, C_i_T has dimension [n_features, n_dmrs_symbols].
        C_i_T = complex_Features[i]
        # To stay consistent with the paper, we need to transpose C_i.
        C_i = tf.transpose(C_i_T)

        # Normalize each row
        row_mean = tf.reduce_mean(C_i, axis=1, keepdims=True)
        F_bar = C_i - row_mean
        F_bar_H = tf.linalg.adjoint(F_bar)
        cov_matrix = tf.matmul(F_bar_H, F_bar) # has dimension [n_features, n_features]

        # Compute with eigh because we are dealing with a Hermitian matrix
        # and this is more efficient than svd
        eigenvalues, U = tf.linalg.eigh(cov_matrix)

        # eigh already sorts eigenvalues (ascending order)
        eigenvalues = tf.reverse(eigenvalues, axis=[0])
        U = tf.reverse(U, axis=[1])

        Sigma_d = eigenvalues[:D_prime]
        U_d = U[:, :D_prime]

        # Prevent negative floats from causing NaN during sqrt
        Sigma_d = tf.maximum(Sigma_d, tf.cast(0.0, Sigma_d.dtype))

        Sigma_d_complex = tf.cast(tf.sqrt(Sigma_d), tf.complex64)
        Z_pca_i = tf.linalg.adjoint(Sigma_d_complex * U_d)

        # Stack real and imaginary parts to stay consistent with existing code
        Z_pca_i = tf.stack([tf.math.real(Z_pca_i), tf.math.imag(Z_pca_i)], axis=-1) # [D_prime, n_features, 2]
        Z_pca_array = Z_pca_array.write(i, Z_pca_i)

    return Z_pca_array.stack()


# The Features tensor has dimension [n_data_samples, n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2]
@tf.function
def dim_red_sammon(Features):
    complex_Features = tf.complex(Features[:, :, :, 0], Features[:, :, :, 1])

    n_samples = tf.shape(Features)[0]
    n_features = tf.shape(Features)[1]
    n_dmrs_symbols = tf.shape(Features)[2]

    Z_sammon_array = tf.TensorArray(tf.complex64, size=n_samples)

    # Iterate over all data samples
    for i in tf.range(n_samples):
        C_i = complex_Features[i]

        C_col = tf.expand_dims(C_i, axis=0) # [1, n_features, n_dmrs_symbols]
        C_row = tf.expand_dims(C_i, axis=1) # [n_features, 1, n_dmrs_symbols]

        D = C_col - C_row # [n_features, n_features, n_dmrs_symbols]
        # Need to use 2-norm and not Frobenius norm because the features are vectors and not matrices
        D_pairwise_distances = tf.norm(D, ord=2, axis=2) # [n_features, n_features]

        