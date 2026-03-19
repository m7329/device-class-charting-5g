import tensorflow as tf

# TODO:
# - D'=2 is set because we want a 2D chart. Will there be cases where we want more than 2 dimensions?
# - Find out what Feature matrices to use for Frobenius norm calculation

# The Features tensor has dimension [n_data_samples, n_tx_ant * n_prbs * 12, n_dmrs_symbols, 2]
# Second argument needs to be a Python integer, not a Tensor!
@tf.function
def dim_red_pca(Features):
    complex_Features = tf.complex(Features[:, :, :, 0], Features[:, :, :, 1])

    # n_features =n_tx_ant * n_prbs * 12
    n_samples = tf.shape(Features)[0]

    # Return PCA scores per DMRS symbol:
    # [n_samples, n_dmrs_symbols, D_prime=2, 2]
    Z_pca_array = tf.TensorArray(tf.float32, size=n_samples)

    # Uncomment this to have averaged DMRS symbols (Studer in his paper with raw 2nd moment)
    edited_features = tf.reduce_mean(complex_Features, axis=2) # [n_samples, n_features]

    # Flatten the features to [n_samples, n_features*n_dmrs_symbols] to apply PCA over each sample
    # edited_features = tf.reshape(complex_Features, [n_samples, -1])

    F = tf.transpose(edited_features) # [x, n_samples]
    row_mean = tf.reduce_mean(F, axis=1, keepdims=True)
    F_bar = F - row_mean
    F_bar_H = tf.linalg.adjoint(F_bar)
    gram_matrix = tf.matmul(F_bar_H, F_bar) # Shape: [n_samples, n_samples]
    eigenvalues, U = tf.linalg.eigh(gram_matrix)

    # Reverse to get descending order (largest eigenvalues first)
    eigenvalues = tf.reverse(eigenvalues, axis=[0])
    U = tf.reverse(U, axis=[1])

    Sigma_d = eigenvalues[:2]
    U_d = U[:, :2] # Shape: [n_samples, D_prime=2]

    Sigma_d_complex = tf.cast(tf.sqrt(Sigma_d), tf.complex64)
    Z_pca = tf.linalg.adjoint(Sigma_d_complex * U_d) # [D_prime=2, n_samples]
    
    Z_pca = tf.transpose(Z_pca) # [n_samples, D_prime=2]
    
    Z_pca_real = tf.math.real(Z_pca)
    Z_pca_imag = tf.math.imag(Z_pca)
    Z_pca_out = tf.stack([Z_pca_real, Z_pca_imag], axis=-1)

    Z_pca_out = tf.cast(Z_pca_out, tf.float32) # [n_samples, D_prime=2, 2]

    # The downstream model expects 4D input
    Z_pca_out = tf.expand_dims(Z_pca_out, axis=2) # [n_samples, D_prime=2, 1, 2]


    '''
    # Need to iterate over first dimension of complex_Features and apply the PCA to each data sample
    for i in tf.range(n_samples):

        # C_i has dimension [n_features, n_dmrs_symbols].
        C_i = complex_Features[i]

        # Normalize each row
        row_mean = tf.reduce_mean(C_i, axis=1, keepdims=True)
        F_bar = C_i - row_mean
        F_bar_H = tf.linalg.adjoint(F_bar)
        cov_matrix = tf.matmul(F_bar_H, F_bar) # has dimension [n_dmrs_symbols, n_dmrs_symbols]
        # This cov_matrix yields the same non-zero eigenvalues as the Gram matrix G = F_bar F_bar^H, which is
        # what we would get if we followed the exact procedure in the paper.

        # Compute with eigh because we are dealing with a Hermitian matrix
        # and this is more efficient than svd
        eigenvalues, U = tf.linalg.eigh(cov_matrix)

        # eigh already sorts eigenvalues (ascending order)
        eigenvalues = tf.reverse(eigenvalues, axis=[0])
        U = tf.reverse(U, axis=[1])

        # Sigma_d is purely real (Hermitian cov_matrix)
        Sigma_d = eigenvalues[:2]
        # U_d is complex
        U_d = U[:, :2]

        # Compute Z as in the paper
        Sigma_d_complex = tf.cast(tf.sqrt(Sigma_d), tf.complex64)
        Z_pca_i = tf.linalg.adjoint(Sigma_d_complex * U_d) # [D_prime=2, n_dmrs_symbols]

        # stack real and imaginary parts into last dimension for Keras-friendly input
        Z_pca_i = tf.stack([tf.math.real(Z_pca_i), tf.math.imag(Z_pca_i)], axis=-1) # [D_prime=2, n_dmrs_symbols, 2]
        Z_pca_i = tf.cast(Z_pca_i, tf.float32)

        Z_pca_array = Z_pca_array.write(i, Z_pca_i)

    return Z_pca_array.stack()

    '''

    return Z_pca_out # [n_samples, D_prime=2, 1, 2]


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

        