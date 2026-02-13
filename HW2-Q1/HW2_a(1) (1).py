"""
Homework 2a — reading A from a .txt file.

Usage:
    - Put your matrix in a text file, e.g. "hw2_data7.txt"
      Each row on its own line, values separated by spaces or tabs.
    - Run: python hw2a_txt_numpy_only.py

Notes:
    - Only depends on numpy and scipy.
    - Implements LU with partial pivoting in pure numpy.
"""

import numpy as np
from scipy.linalg import lu
import time
import sys


# ---------- Utility linear algebra helpers (numpy-only) ----------
def is_symmetric(A, tol=1e-12):
    n,m=np.shape(A)
    if n!=m:
        return False
    return np.allclose(A, A.T, atol=tol, rtol=0)

def is_positive_definite(A):
    # Test via Cholesky (NumPy raises LinAlgError if not PD)
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
    # ---------- Robust forward / backward substitution helpers ----------
def forward_substitution(L, b, tol=1e-16):
    """
    Solve L x = b where L is lower-triangular (may have unit diagonal).
    Returns 1-D x of length = number of columns of L.
    """
    B = np.asarray(b)

    if B.ndim == 1:
        B = B[:, None]         # make (m,1)
    mB, r = B.shape
    m, k = L.shape

    if mB != m:
        raise ValueError(f"Incompatible shapes: L has {m} rows but B has {mB} rows")

    # Prepare output X (k, r)
    X = np.zeros((k, r), dtype=L.dtype)

    # We only have diagonal entries for i = 0..min(m,k)-1
    diag_len = min(m, k)

    for i in range(m):
        if i > 0:
            # L[i, :i] (length min(i,k)) times X[:i, :] (i x r) -> (r,)
            # If i > k then L[i, :i] extends beyond L's columns; slicing safely handles that.
            s = L[i, :i] @ X[:i, :]   # yields shape (r,)
        else:
            s = 0.0

        if i < diag_len:
            diag = L[i, i]
            if abs(diag) <= tol:
                # If diag is numerically 1 (unit-diagonal common in LU), accept it
                if np.isclose(diag, 1.0, atol=1e-12):
                    diag = 1.0
                else:
                    raise np.linalg.LinAlgError(f"Zero (or tiny) diagonal at L[{i},{i}] = {diag}")
            X[i, :] = (B[i, :] - s) / diag
        else:
            # No diagonal element exists for this row (k < i+1). We solve for X entries
            # corresponding to existing columns only; here the row gives an equation that
            # does not include unknowns beyond column k-1. This is an overconstrained row.
            # We'll require that the leftover equation is satisfied exactly (otherwise the system
            # is incompatible). So check B[i,:] - s is (near) zero.
            residual = B[i, :] - s
            if np.max(np.abs(residual)) > 1e-12:
                raise np.linalg.LinAlgError(
                    f"Incompatible/overdetermined system at row {i}: residual {residual}"
                )
            # otherwise nothing to assign to X from this row

    # Return vector if input was vector-like
    return X[:, 0] if X.shape[1] == 1 else X



def backward_substitution(U, b, tol=1e-16):
    """
    Solve U x = b where U is upper-triangular.
    U shape (m,n), b length m. Returns x of length n.
    """
    B = np.asarray(b)

    if B.ndim == 1:
        B = B[:, None]
    mB, r = B.shape
    m, n = U.shape

    if mB != m:
        raise ValueError(f"Incompatible shapes: U has {m} rows but B has {mB} rows")

    X = np.zeros((n, r), dtype=U.dtype)

    # diagonal exists for indices 0..min(m,n)-1
    diag_len = min(m, n)

    # iterate rows from bottom to top
    for row in range(m - 1, -1, -1):
        diag_col = row
        if diag_col >= n:
            # This row contains no diagonal element in U (n < row+1),
            # so the equation is of the form sum_{j=0}^{n-1} U[row,j]*X[j,:] = B[row,:]
            # We can compute s = sum_{j=row+1..n-1} U[row,j] X[j,:] (but here row+1 > n-1)
            # and then check compatibility. Compute full dot to be safe.
            s = U[row, :] @ X[:n, :]   # full row
            residual = B[row, :] - s
            if np.max(np.abs(residual)) > 1e-12:
                raise np.linalg.LinAlgError(
                    f"Incompatible/overdetermined system at row {row}: residual {residual}"
                )
            continue

        # compute contribution of known x's to the right of diagonal
        if diag_col + 1 < n:
            s = U[row, diag_col+1:] @ X[diag_col+1:, :]
        else:
            s = 0.0

        diag = U[row, diag_col]
        if abs(diag) <= tol:
            raise np.linalg.LinAlgError(f"Zero (or tiny) diagonal at U[{row},{diag_col}] = {diag}")

        X[diag_col, :] = (B[row, :] - s) / diag

    return X[:, 0] if X.shape[1] == 1 else X


# ---------- Updated solve_and_compare using forward/backward solves ----------
def _as_2d_if_needed(v):
    """Return v as 2D array (m, r). If v is 1D (m,), return (m,1)."""
    v = np.asarray(v)
    if v.ndim == 1:
        return v[:, None]
    return v

def solve_and_compare(A, b, x_true, methods):
    m, n = A.shape

    # Keep x_true in its provided shape (vector or matrix)
    x_true_arr = np.asarray(x_true)
    # Keep b in its provided shape too
    b_arr = np.asarray(b)

    for method in methods:
        start = time.perf_counter()
        try:
            if method == 'LU':
                P, L, U = lu(A)
                print("shapes:", P.shape, L.shape, U.shape, b_arr.shape)
                Pb = P.T @ b_arr
                y = forward_substitution(L, Pb)
                x = backward_substitution(U, y)

            elif method == 'QR':
                Q, R = np.linalg.qr(A, mode='reduced')
                Qtb = Q.T @ b_arr
                # If R is square (k==n) we can back-substitute, else use lstsq
                if R.shape[0] == R.shape[1]:
                    x = backward_substitution(R, Qtb)
                else:
                    x, *_ = np.linalg.lstsq(R, Qtb, rcond=None)

            elif method == 'Cholesky':
                if m != n:
                    raise ValueError("Cholesky requires a square matrix.")
                L = np.linalg.cholesky(A)
                y = forward_substitution(L, b_arr)
                x = backward_substitution(L.T, y)

            elif method == 'Eigen':
                if m != n:
                    raise ValueError("Eigen decomposition requires a square matrix.")
                if is_symmetric(A):
                    eigvals, V = np.linalg.eigh(A)
                    if np.any(np.isclose(eigvals, 0.0)):
                        raise np.linalg.LinAlgError("Eigen: zero eigenvalue -> singular or ill-conditioned.")
                    # shape-aware division: make eigvals (n,1) to broadcast with V.T @ b
                    Qtb = V.T @ b_arr
                    x = V @ (Qtb / eigvals[:, None])
                else:
                    eigvals, V = np.linalg.eig(A)
                    Vinv_b = np.linalg.solve(V, b_arr)
                    z = Vinv_b / eigvals[:, None]
                    x = V @ z
                    if np.iscomplexobj(x) and np.max(np.abs(np.imag(x))) < 1e-12:
                        x = np.real(x)

            elif method == 'SVD':
                U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
                y = U_svd.T @ b_arr
                tol = max(A.shape) * np.finfo(float).eps * (s.max() if s.size else 1.0)
                s_inv = np.array([1/si if si > tol else 0.0 for si in s])
                z = s_inv[:, None] * y       # ensure (k,1) times (k,r) => (k,r)
                x = Vt.T @ z

            else:
                raise ValueError(f"Unknown method '{method}'")

            elapsed = time.perf_counter() - start

            # ---------- shape-aware error & residual ----------

            # Ensure x is an array
            x = np.asarray(x)

            # Compute error between x and x_true
            # Align shapes: both as 2D arrays (n, r)
            x2 = _as_2d_if_needed(x)
            xt2 = _as_2d_if_needed(x_true_arr)

            if x2.shape != xt2.shape:
                # If ground truth had shape (n,) but solver returned (n,1) or vice versa,
                # try to align by broadcasting single-column to match; otherwise raise.
                if x2.shape[0] == xt2.shape[0] and (x2.shape[1] == 1 or xt2.shape[1] == 1):
                    # broadcast the single column to match other
                    if x2.shape[1] == 1 and xt2.shape[1] > 1:
                        x2 = np.repeat(x2, xt2.shape[1], axis=1)
                    elif xt2.shape[1] == 1 and x2.shape[1] > 1:
                        xt2 = np.repeat(xt2, x2.shape[1], axis=1)
                else:
                    raise ValueError(f"Solver output shape {x2.shape} doesn't match x_true shape {xt2.shape}")

            err = np.linalg.norm(x2 - xt2)

            # Residual: compute A @ x (works for x 1D or 2D if we ensure 2D)
            Ax = A @ x2
            b2 = _as_2d_if_needed(b_arr)
            if Ax.shape != b2.shape:
                raise ValueError(f"Residual shape mismatch: A@x has shape {Ax.shape}, b has shape {b2.shape}")

            res = np.linalg.norm(Ax - b2)

            print(f"{method:<10} | error: {err:.2e} | residual: {res:.2e} | time: {elapsed:.3f}s")

        except Exception as e:
            print(f"{method:<10} | FAILED ({e})")
# ---------- Main entry ----------

def main():
    # Change this filename to your .txt file. Example: "hw2_data7.txt"
    filename = "hw2_data1.txt"

    # Load A from text file. np.loadtxt handles whitespace-separated numeric data.
    try:
        A = np.loadtxt(filename, delimiter=',')
    except OSError:
        print(f"File '{filename}' not found or could not be opened.")
        sys.exit(1)
    except ValueError as ve:
        print(f"Could not parse matrix from '{filename}': {ve}")
        sys.exit(1)

    if A.ndim == 1:
        # Single row — interpret as 1 x n
        A = A.reshape(1, -1)

    m, n = A.shape
    print(f"Matrix is {m} x {n}")

    # ground truth x_true = ones(n)
    x_true = np.ones((n,2))
    b_true = A @ x_true

    # Symmetry / PD checks
    if is_symmetric(A):
        print("A is symmetric.")
        if is_positive_definite(A):
            print("A is SPD!")
        else:
            print("A is NOT positive definite.")
    else:
        print("A is NOT SPD.")

    methods = ['LU', 'QR', 'Cholesky', 'Eigen', 'SVD']
    solve_and_compare(A, b_true, x_true, methods)


if __name__ == "__main__":
    main()
