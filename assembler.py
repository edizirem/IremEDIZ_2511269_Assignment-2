"""
================================================================================
Module      : assembler.py
Purpose     : Assigns global DOF numbers to nodes and assembles the global
              stiffness matrix K and force vector f from element contributions.
              Uses the custom SkylineMatrix for storage.
Inputs      : List of nodes, list of elements, DOFs per node (int)
Outputs     : K (SkylineMatrix), global DOF map, assembled force vector
Assumptions : DOFs are numbered node-by-node: node 1 first, node 2 second, etc.
              Boundary conditions (supports) are applied separately.
Units       : Consistent with element definitions (kN, m)
Author      : CE 4011 Assignment #2
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matrix_lib.skyline_matrix import SkylineMatrix
from matrix_lib.matrix import Matrix


def assign_dofs(nodes: list, dofs_per_node: int) -> int:
    """
    Purpose : Assign sequential global DOF indices to each node.
    Inputs  : nodes        – list of Node objects
              dofs_per_node – int (2 for truss, 3 for frame)
    Outputs : total_dofs (int); modifies node.dof_ids in-place
    """
    dof_counter = 0
    for node in nodes:
        node.dof_ids = list(range(dof_counter, dof_counter + dofs_per_node))
        dof_counter += dofs_per_node
    return dof_counter


def build_skyline_profile(elements: list, total_dofs: int) -> SkylineMatrix:
    """
    Purpose : Build the skyline profile from element DOF connectivity and
              return an empty SkylineMatrix ready for assembly.
    Inputs  : elements   – list of element objects (TrussElement/BeamElement)
              total_dofs – total number of DOFs in the model (int)
    Outputs : SkylineMatrix with profile allocated
    """
    connectivity = [elem.global_dof_indices() for elem in elements]
    K = SkylineMatrix(total_dofs)
    K.build_profile_from_connectivity(connectivity)
    return K


def assemble_stiffness(elements: list, K: SkylineMatrix):
    """
    Purpose : Assemble global stiffness matrix from element stiffness matrices.
    Inputs  : elements – list of element objects
              K        – SkylineMatrix (pre-allocated with correct profile)
    Side-effects: Adds element stiffness contributions into K in-place.
    Algorithm:
        For each element e:
            1. Compute k_e (global element stiffness matrix)
            2. Get global DOF indices dof_e
            3. For each (i,j) in element DOFs:
               K[dof_e[i], dof_e[j]] += k_e[i,j]
    """
    for elem in elements:
        k_e = elem.global_stiffness()
        dof_e = elem.global_dof_indices()
        ndof_e = len(dof_e)
        for i in range(ndof_e):
            for j in range(i, ndof_e):   # upper triangle only (skyline is symmetric)
                gi = dof_e[i]
                gj = dof_e[j]
                val = k_e[i, j]
                if gi <= gj:
                    K.add(gi, gj, val)
                else:
                    K.add(gj, gi, val)


def apply_boundary_conditions(K: SkylineMatrix, f: list,
                               restrained_dofs: list,
                               large_number: float = 1e20):
    """
    Purpose : Apply homogeneous displacement BCs using the penalty method.
              Penalizes the diagonal of restrained DOFs to enforce zero displacement.
    Inputs  : K              – SkylineMatrix (assembled)
              f              – force vector (list of floats, modified in-place)
              restrained_dofs – list of global DOF indices that are fixed (0-based)
              large_number   – penalty value added to diagonal (default 1e20)
    Side-effects: Modifies K and f in-place.
    Notes   : The penalty method is simple and robust; for prescribed
              non-zero displacements a modified RHS term is also needed.
    """
    for dof in restrained_dofs:
        K.add(dof, dof, large_number)
        f[dof] = 0.0  # enforce zero displacement


def apply_prescribed_displacement(K: SkylineMatrix, f: list,
                                   dof: int, value: float,
                                   large_number: float = 1e20):
    """
    Purpose : Apply a prescribed (non-zero) displacement using the penalty method.
    Inputs  : K           – SkylineMatrix
              f           – force vector (modified in-place)
              dof         – global DOF index (0-based)
              value       – prescribed displacement [m or rad]
              large_number – penalty value
    """
    K.add(dof, dof, large_number)
    f[dof] += large_number * value
