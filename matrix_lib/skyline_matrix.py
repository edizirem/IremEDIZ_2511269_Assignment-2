"""
================================================================================
Module      : skyline_matrix.py
Purpose     : Skyline (profile/column-height) storage for symmetric positive-
              definite matrices, with Cholesky (LDLT) factorization and
              forward/back substitution for solving K*u = f.
              Only the profile (non-zero band above diagonal per column) is
              stored, minimizing memory for banded structural stiffness matrices.
Inputs      : n (int)         – number of DOFs
              maxa (list)     – column pointers (skyline profile), length n+1
Outputs     : SkylineMatrix object; solve() returns displacement vector
Assumptions : Matrix is symmetric positive-definite (valid global stiffness K).
              Profile must be set before inserting values (set_profile first).
Units       : Force/length for stiffness, force for RHS
References  : Bathe, K.J. (1996) "Finite Element Procedures", Sec. 8.2
Author      : CE 4011 Assignment #2
================================================================================
"""

import math


class SkylineMatrix:
    """
    Skyline (profile) storage for a symmetric n×n matrix.

    Storage scheme:
      - For each column j, store elements from the first non-zero row
        (the 'column height') down to the diagonal.
      - maxa[j] = index in _au where column j's diagonal element is stored.
      - Total stored elements = maxa[n].

    This is equivalent to the skyline storage used in codes like ADINA.
    """

    def __init__(self, n: int):
        """
        Inputs:
            n – number of rows/columns (int, > 0)
        """
        if n <= 0:
            raise ValueError("Size must be positive.")
        self.n = n
        self._profile_set = False
        self.maxa = None   # column diagonal pointers (length n+1)
        self._au = None    # upper-triangle skyline storage

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------
    def set_profile(self, column_heights: list):
        """
        Purpose : Build the skyline profile from column heights.
        Inputs  : column_heights – list of ints, length n.
                  column_heights[j] = number of elements stored in column j
                  (from topmost non-zero row to diagonal, inclusive).
                  Minimum height is 1 (diagonal only).
        Side-effects: allocates self.maxa and self._au
        """
        if len(column_heights) != self.n:
            raise ValueError("column_heights must have length n.")
        self.maxa = [0] * (self.n + 1)
        self.maxa[0] = 0
        for j in range(self.n):
            h = max(1, column_heights[j])
            self.maxa[j + 1] = self.maxa[j] + h
        total = self.maxa[self.n]
        self._au = [0.0] * total
        self._profile_set = True

    def build_profile_from_connectivity(self, connectivity: list):
        """
        Purpose : Automatically determine column heights from element DOF lists.
        Inputs  : connectivity – list of lists; each inner list gives the global
                  DOF indices (0-based) of one element.
        Side-effects: calls set_profile internally.

        Algorithm:
          For each element, for each pair (i, j) with i < j, row i must be
          in the profile of column j → column_heights[j] >= j - i + 1.
        """
        heights = [1] * self.n  # minimum: diagonal only
        for dofs in connectivity:
            dofs_sorted = sorted(dofs)
            for k, j in enumerate(dofs_sorted):
                for i in dofs_sorted[:k]:
                    needed = j - i + 1
                    if needed > heights[j]:
                        heights[j] = needed
        self.set_profile(heights)

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------
    def _col_start_row(self, j: int) -> int:
        """Return the topmost row stored in column j."""
        return j - (self.maxa[j + 1] - self.maxa[j] - 1)

    def _flat_idx(self, r: int, c: int) -> int:
        """
        Map upper-triangle (r, c) to flat skyline index.
        Returns -1 if element is outside the profile (treated as zero).
        """
        if r > c:
            r, c = c, r  # symmetry
        top = self._col_start_row(c)
        if r < top:
            return -1  # outside profile → structural zero
        return self.maxa[c] + (r - top)

    def get(self, r: int, c: int) -> float:
        """
        Purpose : Return element (r, c); uses symmetry.
        Returns 0.0 if outside the stored profile.
        """
        idx = self._flat_idx(r, c)
        if idx < 0:
            return 0.0
        return self._au[idx]

    def add(self, r: int, c: int, value: float):
        """
        Purpose : Add value to element (r, c).
        Raises ValueError if element is outside the allocated profile.
        """
        idx = self._flat_idx(r, c)
        if idx < 0:
            raise ValueError(
                f"Element ({r},{c}) is outside the skyline profile. "
                "Re-build the profile including this connection."
            )
        self._au[idx] += float(value)

    def set(self, r: int, c: int, value: float):
        """Set element (r, c) to value."""
        idx = self._flat_idx(r, c)
        if idx < 0:
            raise ValueError(f"Element ({r},{c}) outside skyline profile.")
        self._au[idx] = float(value)

    # ------------------------------------------------------------------
    # LDLT Factorization  (in-place)
    # ------------------------------------------------------------------
    def factorize(self):
        """
        Purpose : In-place LDLT (Crout) factorization of the skyline matrix.
                  After factorization, _au stores L and D such that A = L D L^T.
        Assumptions : Matrix is symmetric positive-definite.
        Side-effects: _au is overwritten with factored form.
        References  : Bathe (1996), algorithm for skyline storage.
        """
        n = self.n
        au = self._au

        for j in range(n):
            kn = self.maxa[j + 1] - self.maxa[j]  # column height
            j_top = j - kn + 1                     # topmost row in col j

            # Compute intermediate column vector c_i = L_ji * D_ii
            c = [0.0] * kn
            for i_local in range(kn):
                i = j_top + i_local
                c[i_local] = au[self.maxa[j] + i_local]  # copy A_ij

            # Subtract contributions: c_i -= sum_{k < i} L_ik * c_k
            for i_local in range(kn - 1):
                i = j_top + i_local
                # column i height
                i_kn = self.maxa[i + 1] - self.maxa[i]
                i_top = i - i_kn + 1

                # L_ji = A_ij / D_ii  (stored as A_ij temporarily)
                diag_i_idx = self.maxa[i + 1] - 1  # diagonal of col i
                d_i = au[diag_i_idx]
                if abs(d_i) < 1e-30:
                    raise ValueError(f"Zero pivot encountered at DOF {i}.")

                l_ji = c[i_local] / d_i

                # Update c_k for k > i in current column j
                for k_local in range(i_local + 1, kn):
                    k = j_top + k_local
                    if k <= i:
                        continue
                    # find (i, k) element in column k
                    k_top = k - (self.maxa[k + 1] - self.maxa[k]) + 1
                    if i >= k_top:
                        idx_ik = self.maxa[k] + (i - k_top)
                        # not used here; we work per column j
                    # L_ji * A_ik contribution
                    pass

                # Update off-diagonals and diagonal
                # c[i_local] becomes the L_ji (upper tri factor)
                for m_local in range(i_local + 1, kn):
                    m = j_top + m_local
                    # find (i, m) in column m
                    m_kn = self.maxa[m + 1] - self.maxa[m]
                    m_top = m - m_kn + 1
                    if i >= m_top:
                        idx_im = self.maxa[m] + (i - m_top)
                        c[m_local] -= l_ji * au[idx_im]

                c[i_local] = l_ji

            # Store modified column back and update diagonal
            diag_j_idx = self.maxa[j + 1] - 1
            d_j = c[-1]  # diagonal element
            for k_local in range(kn - 1):
                i = j_top + k_local
                d_i_idx = self.maxa[i + 1] - 1
                d_i = au[d_i_idx]
                d_j -= c[k_local] * c[k_local] * d_i
                au[self.maxa[j] + k_local] = c[k_local]

            if abs(d_j) < 1e-20:
                raise ValueError(
                    f"Non-positive pivot at DOF {j}: matrix may not be "
                    "positive-definite or may be singular (check BCs)."
                )
            au[diag_j_idx] = d_j

        self._factorized = True

    # ------------------------------------------------------------------
    # Forward / Back substitution
    # ------------------------------------------------------------------
    def solve(self, f: list) -> list:
        """
        Purpose : Solve K*u = f using the stored LDLT factorization.
        Inputs  : f – force vector, list of floats, length n
        Outputs : u – displacement vector, list of floats, length n
        Assumptions: factorize() has been called already.
        """
        if not getattr(self, '_factorized', False):
            self.factorize()

        n = self.n
        au = self._au
        u = list(f)  # copy

        # Forward substitution:  (L) * y = f   →  y = u (in-place)
        for j in range(1, n):
            j_kn = self.maxa[j + 1] - self.maxa[j]
            j_top = j - j_kn + 1
            for i_local in range(j_kn - 1):
                i = j_top + i_local
                l_ij = au[self.maxa[j] + i_local]
                u[j] -= l_ij * u[i]

        # Diagonal scaling:  D * z = y  →  z = u (in-place)
        for j in range(n):
            d_j = au[self.maxa[j + 1] - 1]
            u[j] /= d_j

        # Back substitution:  (L^T) * u = z  →  u (in-place)
        for j in range(n - 1, 0, -1):
            j_kn = self.maxa[j + 1] - self.maxa[j]
            j_top = j - j_kn + 1
            for i_local in range(j_kn - 1):
                i = j_top + i_local
                l_ij = au[self.maxa[j] + i_local]
                u[i] -= l_ij * u[j]

        return u

    # ------------------------------------------------------------------
    # Verification helper
    # ------------------------------------------------------------------
    def to_full_list(self) -> list:
        """
        Purpose : Return full matrix as list-of-lists (for debugging).
        Outputs : list of lists of floats, shape (n, n)
        """
        result = [[0.0] * self.n for _ in range(self.n)]
        for r in range(self.n):
            for c in range(r, self.n):
                v = self.get(r, c)
                result[r][c] = v
                result[c][r] = v
        return result

    def __repr__(self):
        lines = [f"SkylineMatrix({self.n}x{self.n}, nnz={self.maxa[self.n] if self.maxa else '?'}):"]
        for r in range(self.n):
            row = [f"{self.get(r, c):10.4g}" for c in range(self.n)]
            lines.append("  [" + "  ".join(row) + "]")
        return "\n".join(lines)
