"""
================================================================================
Package     : matrix_lib
Purpose     : Object-oriented matrix library for structural analysis.
              No numpy or external numerical libraries are used.
Modules:
    matrix           – Dense Matrix class (row-major storage)
    symmetric_matrix – Symmetric Matrix with upper-triangle storage
    skyline_matrix   – Skyline (profile) storage + LDLT solver
    solver           – Dense linear system solver (Gaussian elimination)
Author      : CE 4011 Assignment #2
================================================================================
"""

from .matrix import Matrix
from .symmetric_matrix import SymmetricMatrix
from .skyline_matrix import SkylineMatrix
from .solver import solve_dense, lu_decompose, cholesky_decompose

__all__ = [
    "Matrix",
    "SymmetricMatrix",
    "SkylineMatrix",
    "solve_dense",
    "lu_decompose",
    "cholesky_decompose",
]
