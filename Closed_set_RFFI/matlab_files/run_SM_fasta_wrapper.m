% run SM_fasta.m for a given input and output mat files (for cross-compatibility with Python)

function run_SM_fasta_wrapper(input_mat, output_mat)
S = load(input_mat, 'd_cov', 'no_dims');
d_cov = S.d_cov;
no_dims = S.no_dims;
par = struct();
par.U = size(d_cov, 1);
mappedX = SM_fasta(par, d_cov, no_dims);
save(output_mat, 'mappedX');
end
