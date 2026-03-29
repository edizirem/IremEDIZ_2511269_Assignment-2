"""
================================================================================
Module      : solver.py
Purpose     : Direct linear system solver using Gaussian elimination with
              partial pivoting for dense Matrix objects.
              Also provides LU decomposition utilities.
Inputs      : A – square Matrix (n×n), b – Matrix (n×1) or list
Outputs     : Solution vector x such that A*x = b
Assumptions : A is non-singular; partial pivoting used for numerical stability
Units       : Dimensionless (caller handles physical units)
Author      : CE 4011 Assignment #2
================================================================================
"""

from .matrix import Matrix
import math


def solve_dense(A: Matrix, b) -> list:
    """
    Purpose : Solve A*x = b using Gaussian elimination with partial pivoting.
    Inputs  : A – square Matrix (n×n)
              b – list of floats (length n) or Matrix (n×1)
    Outputs : x – list of floats (length n)
    Raises  : ValueError if A is singular or shape mismatch.
    """
    n = A.rows
    if not A.is_square():
        raise ValueError("Matrix A must be square.")
    if isinstance(b, Matrix):
        rhs = [b[i, 0] for i in range(n)]
    else:
        rhs = list(b)
    if len(rhs) != n:
        raise ValueError("RHS length must equal matrix dimension.")

    # Build augmented matrix [A | b]
    aug = [A._data[r * n:(r + 1) * n] + [rhs[r]] for r in range(n)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-15:
            raise ValueError(f"Singular matrix: zero pivot at column {col}.")
        # Swap rows
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for c in range(col, n + 1):
                aug[row][c] -= factor * aug[col][c]

    # Back substitution
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        x[row] = aug[row][n]
        for c in range(row + 1, n):
            x[row] -= aug[row][c] * x[c]
        x[row] /= aug[row][row]

    return x


def lu_decompose(A: Matrix):
    """
    Purpose : LU decomposition (Doolittle, no pivoting) of square matrix A.
    Inputs  : A – square Matrix (n×n)
    Outputs : (L, U) where L is lower triangular (unit diagonal),
              U is upper triangular, and A = L*U (approximately).
    Assumptions : No row interchanges; may fail for matrices needing pivoting.
    """
    n = A.rows
    L = Matrix.identity(n)
    U = A.copy()

    for k in range(n):
        if abs(U[k, k]) < 1e-15:
            raise ValueError(f"Zero pivot at step {k}; use solve_dense with pivoting.")
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L.set(i, k, factor)
            for j in range(k, n):
                U.set(i, j, U[i, j] - factor * U[k, j])

    return L, U


def cholesky_decompose(A: Matrix):
    """
    Purpose : Cholesky decomposition A = L * L^T for symmetric positive-definite A.
    Inputs  : A – symmetric positive-definite Matrix (n×n)
    Outputs : L – lower triangular Matrix such that L @ L^T == A
    Raises  : ValueError if matrix is not positive-definite.
    """
    n = A.rows
    L = Matrix(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive-definite.")
                L.set(i, j, math.sqrt(s))
            else:
                L.set(i, j, s / L[j, j])
    return L
