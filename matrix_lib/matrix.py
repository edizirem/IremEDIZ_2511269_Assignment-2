"""
================================================================================
Module      : matrix.py
Purpose     : Core dense Matrix class for structural analysis operations
Inputs      : rows (int), cols (int), optional initial data (list of lists)
Outputs     : Matrix object with arithmetic and utility operations
Assumptions : Row-major storage; indices are 0-based internally
Units       : Dimensionless (unit handling is done at the caller level)
Author      : CE 4011 Assignment #2
================================================================================
"""

import math


class Matrix:
    """
    Dense matrix stored as a flat list in row-major order.
    No numpy or external libraries are used.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, rows: int, cols: int, data=None):
        """
        Inputs:
            rows  - number of rows  (int, > 0)
            cols  - number of columns (int, > 0)
            data  - optional list-of-lists [[row0], [row1], ...]
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix dimensions must be positive.")
        self.rows = rows
        self.cols = cols
        if data is not None:
            self._data = [float(data[r][c]) for r in range(rows) for c in range(cols)]
        else:
            self._data = [0.0] * (rows * cols)

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------
    def get(self, r: int, c: int) -> float:
        """Return element at row r, column c (0-based)."""
        return self._data[r * self.cols + c]

    def set(self, r: int, c: int, value: float):
        """Set element at row r, column c (0-based)."""
        self._data[r * self.cols + c] = float(value)

    def __getitem__(self, key):
        r, c = key
        return self.get(r, c)

    def __setitem__(self, key, value):
        r, c = key
        self.set(r, c, value)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: "Matrix") -> "Matrix":
        """
        Matrix addition.
        Inputs : other – Matrix of same shape
        Outputs: new Matrix
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Shape mismatch for addition.")
        result = Matrix(self.rows, self.cols)
        for i in range(len(self._data)):
            result._data[i] = self._data[i] + other._data[i]
        return result

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Matrix subtraction."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Shape mismatch for subtraction.")
        result = Matrix(self.rows, self.cols)
        for i in range(len(self._data)):
            result._data[i] = self._data[i] - other._data[i]
        return result

    def __mul__(self, other):
        """
        Matrix multiplication (Matrix @ Matrix) or scalar multiply.
        Inputs : other – Matrix (n×p) or scalar float/int
        Outputs: new Matrix
        """
        if isinstance(other, (int, float)):
            result = Matrix(self.rows, self.cols)
            result._data = [v * other for v in self._data]
            return result
        if self.cols != other.rows:
            raise ValueError("Incompatible shapes for multiplication.")
        result = Matrix(self.rows, other.cols)
        for r in range(self.rows):
            for c in range(other.cols):
                s = 0.0
                for k in range(self.cols):
                    s += self._data[r * self.cols + k] * other._data[k * other.cols + c]
                result._data[r * other.cols + c] = s
        return result

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        result = Matrix(self.rows, self.cols)
        result._data = [-v for v in self._data]
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def transpose(self) -> "Matrix":
        """
        Purpose : Compute the transpose of this matrix
        Outputs : new Matrix of shape (cols × rows)
        """
        result = Matrix(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                result._data[c * self.rows + r] = self._data[r * self.cols + c]
        return result

    def copy(self) -> "Matrix":
        """Return a deep copy."""
        m = Matrix(self.rows, self.cols)
        m._data = self._data[:]
        return m

    def shape(self):
        return (self.rows, self.cols)

    def norm(self) -> float:
        """Frobenius norm."""
        return math.sqrt(sum(v * v for v in self._data))

    def is_square(self) -> bool:
        return self.rows == self.cols

    def is_symmetric(self, tol=1e-10) -> bool:
        """
        Purpose : Check if matrix is symmetric within tolerance
        Inputs  : tol – absolute tolerance (float)
        """
        if not self.is_square():
            return False
        for r in range(self.rows):
            for c in range(r + 1, self.cols):
                if abs(self.get(r, c) - self.get(c, r)) > tol:
                    return False
        return True

    def to_list(self):
        """Return list-of-lists representation."""
        return [[self._data[r * self.cols + c] for c in range(self.cols)]
                for r in range(self.rows)]

    # ------------------------------------------------------------------
    # Static constructors
    # ------------------------------------------------------------------
    @staticmethod
    def identity(n: int) -> "Matrix":
        """n×n identity matrix."""
        m = Matrix(n, n)
        for i in range(n):
            m.set(i, i, 1.0)
        return m

    @staticmethod
    def zeros(rows: int, cols: int) -> "Matrix":
        return Matrix(rows, cols)

    @staticmethod
    def from_list(data) -> "Matrix":
        """
        Create Matrix from list of lists.
        Inputs : data – list of lists of numbers
        """
        rows = len(data)
        cols = len(data[0])
        return Matrix(rows, cols, data)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __repr__(self):
        lines = []
        for r in range(self.rows):
            row_vals = [f"{self._data[r * self.cols + c]:12.6g}" for c in range(self.cols)]
            lines.append("  [" + "  ".join(row_vals) + "]")
        return f"Matrix({self.rows}x{self.cols}):\n" + "\n".join(lines)
