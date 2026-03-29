"""
================================================================================
Module      : elements.py
Purpose     : Data structures for nodes and 2D truss/beam elements.
              Defines Node, TrussElement, and BeamElement classes.
              Computes local and global element stiffness matrices.
Inputs      : Node coordinates, element connectivity, material/section properties
Outputs     : Element stiffness matrices (Matrix objects, no numpy)
Assumptions : 2D plane frame (truss + Euler-Bernoulli beam elements)
              Coordinate system: X horizontal, Y vertical
Units       : Length in meters [m], Force in kN, Stress in kPa
Author      : CE 4011 Assignment #2
================================================================================
"""

import math
from matrix_lib.matrix import Matrix


class Node:
    """
    Represents a structural node.

    Attributes:
        node_id  – unique integer identifier (1-based)
        x, y     – coordinates [m]
        dof_ids  – list of global DOF indices assigned (set during assembly)
    """

    def __init__(self, node_id: int, x: float, y: float):
        """
        Inputs:
            node_id – unique node number (int)
            x       – X coordinate [m]
            y       – Y coordinate [m]
        """
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.dof_ids = []  # assigned during DOF numbering

    def __repr__(self):
        return f"Node({self.node_id}: x={self.x}, y={self.y}, dofs={self.dof_ids})"


class TrussElement:
    """
    2D truss (bar) element with 2 nodes and 4 DOFs (ux1, uy1, ux2, uy2).

    Assumptions:
        - Axial deformation only; no bending.
        - Uniform cross-section and material properties.
    """

    NDOF_PER_NODE = 2
    NDOF = 4  # total element DOFs

    def __init__(self, elem_id: int, node_i: Node, node_j: Node,
                 E: float, A: float):
        """
        Inputs:
            elem_id – element number (int)
            node_i  – start Node object
            node_j  – end Node object
            E       – Young's modulus [kN/m²]
            A       – cross-sectional area [m²]
        """
        self.elem_id = elem_id
        self.node_i = node_i
        self.node_j = node_j
        self.E = float(E)
        self.A = float(A)
        self._compute_geometry()

    def _compute_geometry(self):
        """
        Purpose : Compute element length and direction cosines.
        Outputs : self.L (length), self.cx (cos θ), self.cy (sin θ)
        """
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        self.L = math.sqrt(dx * dx + dy * dy)
        if self.L < 1e-12:
            raise ValueError(f"Element {self.elem_id} has zero length.")
        self.cx = dx / self.L  # cos(θ)
        self.cy = dy / self.L  # sin(θ)

    def local_stiffness(self) -> Matrix:
        """
        Purpose : Compute 2×2 local stiffness matrix in local (axial) coords.
        Outputs : Matrix (2×2)
        Formula : k_local = (EA/L) * [[1, -1], [-1, 1]]
        """
        ea_l = self.E * self.A / self.L
        k = Matrix(2, 2, [[ea_l, -ea_l], [-ea_l, ea_l]])
        return k

    def transformation_matrix(self) -> Matrix:
        """
        Purpose : 2×4 transformation matrix from global to local DOFs.
        Outputs : Matrix (2×4)
                  T = [[cx, cy, 0,  0 ],
                       [0,  0,  cx, cy]]
        """
        cx, cy = self.cx, self.cy
        T = Matrix(2, 4, [[cx, cy, 0.0, 0.0],
                           [0.0, 0.0, cx, cy]])
        return T

    def global_stiffness(self) -> Matrix:
        """
        Purpose : Compute 4×4 global element stiffness matrix.
        Outputs : Matrix (4×4)
        Formula : k_global = T^T * k_local * T
        """
        k_loc = self.local_stiffness()
        T = self.transformation_matrix()
        Tt = T.transpose()
        k_glob = Tt * k_loc * T
        return k_glob

    def global_dof_indices(self) -> list:
        """
        Purpose : Return list of global DOF indices for this element.
        Outputs : list of 4 ints [u_i_x, u_i_y, u_j_x, u_j_y]
        """
        return self.node_i.dof_ids + self.node_j.dof_ids

    def axial_force(self, u_global: list) -> float:
        """
        Purpose : Compute axial force in element from global displacements.
        Inputs  : u_global – full displacement vector (list of floats)
        Outputs : N – axial force [kN] (positive = tension)
        """
        dof = self.global_dof_indices()
        u_e = [u_global[d] for d in dof]
        T = self.transformation_matrix()
        u_loc = [sum(T[i, j] * u_e[j] for j in range(4)) for i in range(2)]
        N = self.E * self.A / self.L * (u_loc[1] - u_loc[0])
        return N

    def __repr__(self):
        return (f"TrussElement({self.elem_id}: "
                f"N{self.node_i.node_id}-N{self.node_j.node_id}, "
                f"L={self.L:.4f}m, E={self.E:.2e}, A={self.A:.4f})")


class BeamElement:
    """
    2D Euler-Bernoulli beam-column element with 2 nodes and 6 DOFs.
    DOFs per node: (ux, uy, rz) → 3 DOFs/node, 6 DOFs/element.

    Assumptions:
        - Euler-Bernoulli (plane sections remain plane, no shear deformation).
        - Uniform cross-section along element length.
        - Combined axial + bending behavior.
    """

    NDOF_PER_NODE = 3
    NDOF = 6

    def __init__(self, elem_id: int, node_i: Node, node_j: Node,
                 E: float, A: float, I: float):
        """
        Inputs:
            elem_id – element number (int)
            node_i  – start Node
            node_j  – end Node
            E       – Young's modulus [kN/m²]
            A       – cross-sectional area [m²]
            I       – second moment of area [m⁴]
        """
        self.elem_id = elem_id
        self.node_i = node_i
        self.node_j = node_j
        self.E = float(E)
        self.A = float(A)
        self.I = float(I)
        self._compute_geometry()

    def _compute_geometry(self):
        """Compute length and direction cosines."""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        self.L = math.sqrt(dx * dx + dy * dy)
        if self.L < 1e-12:
            raise ValueError(f"Element {self.elem_id} has zero length.")
        self.cx = dx / self.L
        self.cy = dy / self.L

    def local_stiffness(self) -> Matrix:
        """
        Purpose : 6×6 local stiffness matrix in element local axes.
        Outputs : Matrix (6×6)
        Notation: DOFs = [u1, v1, θ1, u2, v2, θ2] in local coords
        Reference: McGuire et al., Matrix Structural Analysis, 2nd ed.
        """
        E, A, I, L = self.E, self.A, self.I, self.L
        ea = E * A / L
        ei2 = 2.0 * E * I / L
        ei4 = 4.0 * E * I / L
        ei6 = 6.0 * E * I / (L * L)
        ei12 = 12.0 * E * I / (L * L * L)

        data = [
            [ ea,    0,    0,   -ea,    0,    0],
            [  0,  ei12,  ei6,    0, -ei12,  ei6],
            [  0,  ei6,  ei4,    0,  -ei6,  ei2],
            [-ea,    0,    0,    ea,    0,    0],
            [  0, -ei12, -ei6,    0,  ei12, -ei6],
            [  0,  ei6,  ei2,    0,  -ei6,  ei4],
        ]
        return Matrix(6, 6, data)

    def transformation_matrix(self) -> Matrix:
        """
        Purpose : 6×6 transformation matrix from local to global coords.
        Outputs : Matrix (6×6)
        """
        cx, cy = self.cx, self.cy
        sx = -cy
        sy =  cx
        T_data = [
            [cx,  cy, 0, 0,  0, 0],
            [sx,  sy, 0, 0,  0, 0],
            [ 0,   0, 1, 0,  0, 0],
            [ 0,   0, 0, cx, cy, 0],
            [ 0,   0, 0, sx, sy, 0],
            [ 0,   0, 0, 0,  0,  1],
        ]
        return Matrix(6, 6, T_data)

    def global_stiffness(self) -> Matrix:
        """
        Purpose : 6×6 global stiffness matrix: k = T^T * k_local * T
        Outputs : Matrix (6×6)
        """
        k_loc = self.local_stiffness()
        T = self.transformation_matrix()
        Tt = T.transpose()
        return Tt * k_loc * T

    def global_dof_indices(self) -> list:
        """
        Purpose : Return list of 6 global DOF indices.
        Outputs : [ux_i, uy_i, rz_i, ux_j, uy_j, rz_j]
        """
        return self.node_i.dof_ids + self.node_j.dof_ids

    def local_forces(self, u_global: list) -> list:
        """
        Purpose : Compute local element forces from global displacements.
        Inputs  : u_global – full displacement vector
        Outputs : list of 6 local forces [N1, V1, M1, N2, V2, M2]
        """
        dof = self.global_dof_indices()
        u_e = [u_global[d] for d in dof]
        T = self.transformation_matrix()
        u_loc = [sum(T[i, j] * u_e[j] for j in range(6)) for i in range(6)]
        k_loc = self.local_stiffness()
        f_loc = [sum(k_loc[i, j] * u_loc[j] for j in range(6)) for i in range(6)]
        return f_loc

    def __repr__(self):
        return (f"BeamElement({self.elem_id}: "
                f"N{self.node_i.node_id}-N{self.node_j.node_id}, "
                f"L={self.L:.4f}m, E={self.E:.2e}, A={self.A:.4f}, I={self.I:.4e})")
