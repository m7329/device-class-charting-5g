import tensorflow as tf

# The Features tensor has dimension [n_data_samples, n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2]
@tf.function
dim_red_pca(Features, D_prime=2):
    complex_Features = tf.complex(Features[:, :, :, 0], Features[:, :, :, 1])

    n_samples = tf.shape(Features)[0]

    Z_pca_array = tf.TensorArray(tf.complex64, size=n_samples)

    # Need to iterate over first dimension of complex_Features and apply the PCA to each data sample
    for i in tf.range(n_samples):
        C_i = complex_Features[i]

        row_mean = tf.reduce_mean(C_i, axis=1, keepdims=True)
        F_bar = C_i - row_mean
        F_bar_H = tf.linalg.adjoint(F_bar)
        cov_matrix = tf.matmul(F_bar_H, F_bar)

        # Compute with eigh because we are dealing with a Hermitian matrix
        # and this is more efficient than svd
        eigenvalues, U = tf.linalg.eigh(cov_matrix)

        # eigh already sorts eigenvalues (ascending order)
        eigenvalues = tf.reverse(eigenvalues, axis=[0])
        U = tf.reverse(U, axis=[1])

        Sigma_d = eigenvalues[:D_prime]
        U_d = U[:, :D_prime]

        # Prevent negative floats from causing NaN during sqrt
        Sigma_d = tf.maximum(Sigma_d, 0.0)

        Sigma_d_complex = tf.cast(tf.sqrt(Sigma_d), tf.complex64)
        Z_pca_i = tf.linalg.adjoint(Sigma_d_complex * U_d)

        Z_pca_array = Z_pca_array.write(i, Z_pca_i)

    return Z_pca_array.stack()


# The Features tensor has dimension [n_data_samples, n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2]
@tf.function
dim_red_sammon(Features):
    complex_Features = tf.complex(Features[:, :, :, 0], Features[:, :, :, 1])

    n_samples = tf.shape(Features)[0]

    D_pairwise_distances = tf.TensorArray(tf.float32, size=n_samples)

    Z_sammon_array = tf.TensorArray(tf.complex64, size=n_samples)

    # Iterate over all data samples
    for i in tf.range(n_samples):
        C_i = complex_Features[i]
        D_pairwise_distances = tf.norm(C_i[:,1:] - C_i[:,:-1], ord='fro',axis=1, keepdims=False)