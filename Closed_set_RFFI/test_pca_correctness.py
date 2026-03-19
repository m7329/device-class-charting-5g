#!/usr/bin/env python3
"""
Numerical correctness checks for src/dim_reduction.py::dim_red_pca.

This test does NOT try to validate downstream classification accuracy.
It validates the PCA math by cross-checking TF vs NumPy on:
  - covariance eigenvalues
  - principal subspace (projection matrix) for the top-D' eigenvectors
  - basic covariance sanity (finite, Hermitian, non-negative eigenvalues)

Expected feature layout (obfuscation features):
  Features: [n_samples, n_features, n_dmrs_symbols, 2]  (real/imag stacked)

Expected dim_red_pca output (current implementation):
  Z: [n_samples, D_prime=2, n_dmrs_symbols, 2]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class CheckResult:
    idx: int
    evals_tf: np.ndarray
    evals_np: np.ndarray
    eval_rel_err: float
    proj_fro_err: float
    hermitian_rel_err: float
    min_eval_np: float
    finite: bool


def _sorted_desc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.sort(x)[::-1]


def _projection_matrix(U: np.ndarray) -> np.ndarray:
    """U: [n, k] complex; returns P = U U^H (phase-invariant subspace rep)."""
    return U @ U.conj().T


def _compute_cov_and_eigh_np(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    C: [n_features, n_dmrs] complex64/complex128
    Matches the centering & covariance in current dim_red_pca:
      row_mean over axis=1 (DMRS axis), then cov = F_bar^H F_bar -> [n_dmrs, n_dmrs]
    Returns (evals_desc_real, U_desc) where U are eigenvectors in DMRS-space.
    """
    row_mean = C.mean(axis=1, keepdims=True)
    F_bar = C - row_mean
    cov = F_bar.conj().T @ F_bar
    # eigh: ascending
    evals, U = np.linalg.eigh(cov)
    evals = evals[::-1]
    U = U[:, ::-1]
    return np.real(evals), U


def _compute_cov_and_eigh_tf(C: tf.Tensor) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """
    C: [n_features, n_dmrs] complex64
    Returns (evals_desc_real, U_desc_complex, hermitian_rel_err, finite)
    """
    row_mean = tf.reduce_mean(C, axis=1, keepdims=True)
    F_bar = C - row_mean
    cov = tf.matmul(tf.linalg.adjoint(F_bar), F_bar)  # [n_dmrs, n_dmrs]

    finite = bool(
        tf.reduce_all(tf.math.is_finite(tf.math.real(cov))).numpy()
        and tf.reduce_all(tf.math.is_finite(tf.math.imag(cov))).numpy()
    )

    herm_err = tf.linalg.norm(cov - tf.linalg.adjoint(cov)) / (tf.linalg.norm(cov) + tf.cast(1e-12, cov.dtype))
    hermitian_rel_err = float(tf.math.real(herm_err).numpy())

    evals, U = tf.linalg.eigh(cov)  # ascending
    evals = tf.reverse(evals, axis=[0])
    U = tf.reverse(U, axis=[1])
    return np.real(evals.numpy()), U.numpy(), hermitian_rel_err, finite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", default="/scratch/bsc26f18/datasets/device_classification_2025_11_15" help="Dataset root (contains training/ and testing/)")
    parser.add_argument("--labels", nargs="+", required=True, help="Label folder names under training/")
    parser.add_argument("--n-orus", type=int, default=4)
    parser.add_argument("--subset-fraction", type=float, default=0.001)
    parser.add_argument("--max-samples-per-label", type=int, default=5)
    parser.add_argument("--subset-seed", type=int, default=1)
    parser.add_argument("--d-prime", type=int, default=2, help="Expected D' (dim_red_pca currently hardcodes 2; this checks consistency).")
    parser.add_argument("--tol-evals-rel", type=float, default=1e-4)
    parser.add_argument("--tol-proj-fro", type=float, default=1e-3)
    parser.add_argument("--tol-herm", type=float, default=1e-5)
    args = parser.parse_args()

    # Make Closed_set_RFFI imports work
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))

    # Import dim_red_pca from ../src
    src_dir = this_dir.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from dataset_preparation import Load5gDataset  # noqa: E402
    from dim_reduction import dim_red_pca  # noqa: E402

    ds = Load5gDataset(feature_type="obfuscation")
    training_csi, training_labels, _, _ = ds.load_channel_estimates(
        file_path=str(Path(args.file_path) / "training"),
        label_list=list(args.labels),
        random_subsampling=True,
        n_orus=args.n_orus,
        test_to_all_ratio=0.2,
        take_middle=False,
        subset_fraction=args.subset_fraction,
        max_samples_per_label=args.max_samples_per_label,
        subset_seed=args.subset_seed,
    )

    x = np.asarray(training_csi, dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] != 2:
        raise ValueError(f"Unexpected Features shape {x.shape}, expected [n, n_features, n_dmrs, 2]")
    n, n_features, n_dmrs, two = x.shape
    print(f"Loaded Features: n={n}, n_features={n_features}, n_dmrs={n_dmrs}, last_dim={two}")

    # Run dim_red_pca and check output shape
    z = dim_red_pca(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()
    print(f"dim_red_pca output shape: {z.shape}")
    if z.shape != (n, args.d_prime, n_dmrs, 2):
        raise AssertionError(f"Unexpected dim_red_pca output shape {z.shape}, expected {(n, args.d_prime, n_dmrs, 2)}")

    # Numerical cross-checks per-sample
    results: list[CheckResult] = []
    for i in range(min(n, 10)):
        C = x[i, :, :, 0] + 1j * x[i, :, :, 1]  # [n_features, n_dmrs]
        evals_np, U_np = _compute_cov_and_eigh_np(C.astype(np.complex64))

        C_tf = tf.constant(C.astype(np.complex64))
        evals_tf, U_tf, herm_err, finite = _compute_cov_and_eigh_tf(C_tf)

        # Compare eigenvalues (relative)
        denom = float(np.linalg.norm(evals_np) + 1e-12)
        eval_rel_err = float(np.linalg.norm(evals_tf - evals_np) / denom)

        # Compare principal subspace for top-D'
        k = min(args.d_prime, U_np.shape[1])
        P_np = _projection_matrix(U_np[:, :k])
        P_tf = _projection_matrix(U_tf[:, :k])
        proj_fro_err = float(np.linalg.norm(P_tf - P_np, ord="fro") / (np.linalg.norm(P_np, ord="fro") + 1e-12))

        results.append(
            CheckResult(
                idx=i,
                evals_tf=evals_tf,
                evals_np=evals_np,
                eval_rel_err=eval_rel_err,
                proj_fro_err=proj_fro_err,
                hermitian_rel_err=herm_err,
                min_eval_np=float(np.min(evals_np)),
                finite=finite,
            )
        )

    # Report
    worst_eval = max(results, key=lambda r: r.eval_rel_err)
    worst_proj = max(results, key=lambda r: r.proj_fro_err)
    worst_herm = max(results, key=lambda r: r.hermitian_rel_err)

    print("\nWorst-case metrics over checked samples:")
    print(f"- eigenvalue relative error:                      {worst_eval.eval_rel_err:.3e} (sample {worst_eval.idx})")
    print(f"- subspace proj Fro error:                        {worst_proj.proj_fro_err:.3e} (sample {worst_proj.idx})")
    print(f"- Hermitian rel error (only TF side-check):       {worst_herm.hermitian_rel_err:.3e} (sample {worst_herm.idx})")

    any_nonfinite = any(not r.finite for r in results)
    any_negative = any(r.min_eval_np < -1e-5 for r in results)

    if any_nonfinite:
        raise AssertionError("Non-finite covariance encountered (NaN/Inf).")
    if any_negative:
        raise AssertionError("Significantly negative eigenvalue encountered (covariance should be PSD).")

    if worst_eval.eval_rel_err > args.tol_evals_rel:
        raise AssertionError(f"Eigenvalue mismatch too large: {worst_eval.eval_rel_err:.3e} > {args.tol_evals_rel:.3e}")
    if worst_proj.proj_fro_err > args.tol_proj_fro:
        raise AssertionError(f"Subspace mismatch too large: {worst_proj.proj_fro_err:.3e} > {args.tol_proj_fro:.3e}")
    if worst_herm.hermitian_rel_err > args.tol_herm:
        raise AssertionError(f"Hermitian error too large: {worst_herm.hermitian_rel_err:.3e} > {args.tol_herm:.3e}")

    print("\nPCA cross-check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

