"""
================================================================================
Module      : symmetric_matrix.py
Purpose     : Symmetric matrix storing only the upper triangle.
              Storage: n*(n+1)/2 elements (roughly half of full storage).
              Suitable for symmetric stiffness matrices in structural analysis.
Inputs      : n (int) – size of the square symmetric matrix
Outputs     : SymmetricMatrix object
Assumptions : Matrix is square and symmetric; only upper triangle is stored.
Units       : Dimensionless
Author      : CE 4011 Assignment #2
================================================================================
"""

import math
from .matrix import Matrix


class SymmetricMatrix:
    """
    Symmetric n×n matrix stored in upper-triangular packed form.
    Element (i,j) with i<=j is stored at index i*n - i*(i-1)//2 + (j-i).
    This halves the memory compared to full storage.
    """

    def __init__(self, n: int):
        """
        Inputs:
            n – matrix dimension (int, > 0)
        """
        if n <= 0:
            raise ValueError("Size must be positive.")
        self.n = n
        # Upper triangle including diagonal: n*(n+1)//2 elements
        self._data = [0.0] * (n * (n + 1) // 2)

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------
    def _idx(self, r: int, c: int) -> int:
        """
        Purpose : Map (row, col) to flat upper-triangle index.
        Assumes  : r <= c (swap if needed before calling).
        """
        if r > c:
            r, c = c, r  # enforce upper triangle
        return r * self.n - r * (r - 1) // 2 + (c - r)

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------
    def get(self, r: int, c: int) -> float:
        """Return element (r, c); exploits symmetry."""
        return self._data[self._idx(r, c)]

    def set(self, r: int, c: int, value: float):
        """Set element (r, c); automatically keeps symmetry."""
        self._data[self._idx(r, c)] = float(value)

    def add(self, r: int, c: int, value: float):
        """Add value to element (r, c)."""
        self._data[self._idx(r, c)] += float(value)

    def __getitem__(self, key):
        return self.get(*key)

    def __setitem__(self, key, value):
        self.set(*key, value)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def to_matrix(self) -> Matrix:
        """
        Purpose : Convert to a full dense Matrix (for verification/output).
        Outputs : Matrix of shape (n, n)
        """
        m = Matrix(self.n, self.n)
        for r in range(self.n):
            for c in range(r, self.n):
                v = self.get(r, c)
                m.set(r, c, v)
                m.set(c, r, v)
        return m

    def to_list(self):
        return self.to_matrix().to_list()

    # ------------------------------------------------------------------
    # Matrix–vector product
    # ------------------------------------------------------------------
    def multiply_vector(self, vec: list) -> list:
        """
        Purpose : Compute A*x where A is this symmetric matrix.
        Inputs  : vec – list of floats, length n
        Outputs : list of floats, length n
        """
        if len(vec) != self.n:
            raise ValueError("Vector length mismatch.")
        result = [0.0] * self.n
        for r in range(self.n):
            for c in range(self.n):
                result[r] += self.get(r, c) * vec[c]
        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __repr__(self):
        lines = [f"SymmetricMatrix({self.n}x{self.n}):"]
        for r in range(self.n):
            row = [f"{self.get(r, c):12.6g}" for c in range(self.n)]
            lines.append("  [" + "  ".join(row) + "]")
        return "\n".join(lines)
