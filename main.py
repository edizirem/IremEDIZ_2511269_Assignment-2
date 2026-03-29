"""
================================================================================
Module      : main.py
Purpose     : 2D Frame Analysis Program following the algorithm described in
              "Frame Analysis Program" course document.
              Phases: Input -> Equation Numbering -> Global K -> Global F ->
                      Solve -> Member End Forces -> Output
Usage       : python main.py
Units       : kN [force], m [length], kN/m^2 [modulus]
Author      : CE 4011 Assignment #2
================================================================================
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matrix_lib.matrix import Matrix
from matrix_lib.skyline_matrix import SkylineMatrix
from matrix_lib.solver import solve_dense


# ==============================================================================
# A. INPUT PHASE
# ==============================================================================

def input_phase():
    """
    Purpose  : Define all model data following the document array conventions.
    Outputs  : NumNode, NumElem, NumSupport, NumLoadJoint, XY, M, C, S, L
    Arrays:
        XY[i]   = [X, Y]               nodal coordinates (0-based)
        M[i]    = [A, I, E]            material/section property set i
        C[i]    = [start, end, matID]  element connectivity (1-based node IDs)
        S[i]    = [nodeID, rx, ry, rz] support BC (1=restrained, 0=free)
        L[i]    = [nodeID, Fx, Fy, Mz] applied nodal loads
    """
    NumNode      = 4
    NumElem      = 4
    NumSupport   = 2
    NumLoadJoint = 2

    # XY: nodal coordinates [X, Y] in metres
    # Node 1:(4,0)  Node 2:(4,3)  Node 3:(0,3)  Node 4:(0,0)
    XY = [
        [4.0, 0.0],
        [4.0, 3.0],
        [0.0, 3.0],
        [0.0, 0.0],
    ]

    # M: material sets [A, I, E]
    M = [
        [0.02, 0.08, 200000.0],   # set 1  members 1-3
        [0.01, 0.01, 200000.0],   # set 2  member 4
    ]

    # C: connectivity [start_node, end_node, mat_set] (1-based)
    C = [
        [1, 2, 1],
        [2, 3, 1],
        [4, 3, 1],
        [1, 3, 2],
    ]

    # S: supports [nodeID, ux, uy, rz]  (1=restrained)
    S = [
        [1, 1, 1, 0],
        [4, 0, 1, 0],
    ]

    # L: loads [nodeID, Fx, Fy, Mz]
    L = [
        [2,  10.0, -10.0, 0.0],
        [3, -10.0, -10.0, 0.0],
    ]

    return NumNode, NumElem, NumSupport, NumLoadJoint, XY, M, C, S, L


# ==============================================================================
# B. EQUATION NUMBERING
# ==============================================================================

def equation_numbering(NumNode, S):
    """
    Purpose  : Build equation number array E (Step 7 of the document).
               E[node][dof] = global eq number (1-based), 0 if restrained.
    Inputs   : NumNode, S (support array)
    Outputs  : E (NumNode x 3 list), NumEq (total active DOFs)
    """
    E = [[0, 0, 0] for _ in range(NumNode)]

    # Mark restrained DOFs
    for sup in S:
        node_id = sup[0] - 1
        for dof in range(3):
            if sup[dof + 1] == 1:
                E[node_id][dof] = -1

    # Assign consecutive equation numbers to free DOFs
    eq_num = 0
    for node in range(NumNode):
        for dof in range(3):
            if E[node][dof] == 0:
                eq_num += 1
                E[node][dof] = eq_num
            else:
                E[node][dof] = 0

    return E, eq_num


# ==============================================================================
# C. GLOBAL STIFFNESS MATRIX [K]
# ==============================================================================

def element_local_stiffness(A, I, E_mod, L):
    """
    Purpose  : 6x6 local stiffness matrix (Step 9.1.1).
    DOF order: [u1, v1, theta1, u2, v2, theta2]
    """
    ea   = E_mod * A / L
    ei2  = 2.0 * E_mod * I / L
    ei4  = 4.0 * E_mod * I / L
    ei6  = 6.0 * E_mod * I / (L * L)
    ei12 = 12.0 * E_mod * I / (L * L * L)

    data = [
        [ ea,    0,    0,  -ea,    0,    0],
        [  0, ei12,  ei6,    0,-ei12,  ei6],
        [  0,  ei6,  ei4,    0,  -ei6, ei2],
        [-ea,    0,    0,   ea,    0,    0],
        [  0,-ei12, -ei6,    0, ei12, -ei6],
        [  0,  ei6,  ei2,    0,  -ei6, ei4],
    ]
    return Matrix(6, 6, data)


def element_rotation_matrix(cx, cy):
    """
    Purpose  : 6x6 rotation matrix R (Step 9.1.2).
    """
    sx = -cy
    sy =  cx
    data = [
        [cx, cy, 0,  0,  0, 0],
        [sx, sy, 0,  0,  0, 0],
        [ 0,  0, 1,  0,  0, 0],
        [ 0,  0, 0, cx, cy, 0],
        [ 0,  0, 0, sx, sy, 0],
        [ 0,  0, 0,  0,  0, 1],
    ]
    return Matrix(6, 6, data)


def element_global_stiffness(A, I, E_mod, xi, yi, xj, yj):
    """
    Purpose  : k_global = R^T * k_local * R  (Step 9.1.3).
    Outputs  : k_global (Matrix 6x6), L (element length)
    """
    dx = xj - xi
    dy = yj - yi
    L  = math.sqrt(dx*dx + dy*dy)
    cx = dx / L
    cy = dy / L

    k_loc = element_local_stiffness(A, I, E_mod, L)
    R     = element_rotation_matrix(cx, cy)
    return R.transpose() * k_loc * R, L


def build_global_stiffness(NumEq, NumElem, XY, M, C, E_arr):
    """
    Purpose  : Assemble K by looping over elements (Steps 9.2, 10).
               Discards any term where destination equation number is 0.
    Outputs  : K – Matrix (NumEq x NumEq)
    """
    K = Matrix(NumEq, NumEq)

    for i in range(NumElem):
        sn    = C[i][0] - 1
        en    = C[i][1] - 1
        mid   = C[i][2] - 1
        A     = M[mid][0]
        I_val = M[mid][1]
        E_mod = M[mid][2]

        k_g, _ = element_global_stiffness(A, I_val, E_mod,
                                           XY[sn][0], XY[sn][1],
                                           XY[en][0], XY[en][1])

        # Auxiliary DOF vector G (Step 10)
        G = E_arr[sn][:] + E_arr[en][:]

        for p in range(6):
            P = G[p]
            if P == 0:
                continue
            for q in range(6):
                Q = G[q]
                if Q == 0:
                    continue
                K.set(P-1, Q-1, K.get(P-1, Q-1) + k_g.get(p, q))

    return K


# ==============================================================================
# D. GLOBAL LOAD VECTOR [F]
# ==============================================================================

def build_load_vector(NumEq, NumLoadJoint, L_loads, E_arr):
    """
    Purpose  : Assemble F by looping over loaded joints (Step 11).
    Outputs  : F – list of floats (length NumEq)
    """
    F = [0.0] * NumEq

    for i in range(NumLoadJoint):
        node_id = L_loads[i][0] - 1
        for q in range(3):
            Q = E_arr[node_id][q]
            if Q != 0:
                F[Q - 1] += L_loads[i][q + 1]

    return F


# ==============================================================================
# E. STRUCTURAL DISPLACEMENTS
# ==============================================================================

def solve_displacements(K, F, connectivity=None):
    """
    Purpose  : Solve [K]{D} = {F} (Step 12).
               Primary solver: dense Gaussian elimination (solver.py).
               Verification: skyline LDLT cross-check.
    Inputs   : K            – Matrix (NumEq x NumEq)
               F            – list of floats (length NumEq)
               connectivity – list of active DOF lists per element (for skyline profile)
    Outputs  : D – list of floats (length NumEq)
    Notes    : solve_dense from solver.py is the primary solver here because it
               uses Gaussian elimination with partial pivoting — robust for any
               DOF ordering. The skyline solver serves as the cross-check.
    """
    n = K.rows

    # Primary solve: dense Gaussian elimination (solver.py)
    D_dense = solve_dense(K, F)

    # Cross-check: skyline LDLT
    if connectivity is not None:
        sky = SkylineMatrix(n)
        sky.build_profile_from_connectivity(connectivity)
        for r in range(n):
            for c in range(r, n):
                v = K.get(r, c)
                if abs(v) > 1e-20:
                    sky.add(r, c, v)
        sky.factorize()
        D_sky = sky.solve(F)
        max_diff = max(abs(D_dense[i] - D_sky[i]) for i in range(n))
        print(f"  [Verification] Dense vs Skyline max diff: {max_diff:.2e}  "
              + ("PASSED" if max_diff < 1e-4 else "WARNING — check DOF ordering"))

    return D_dense


# ==============================================================================
# F. MEMBER END FORCES
# ==============================================================================

def member_end_forces(NumElem, XY, M, C, E_arr, D):
    """
    Purpose  : Compute local member end forces for each element (Steps 13.1-13.5).
    Outputs  : list of 6-element lists [N1, V1, M1, N2, V2, M2] per element
    """
    all_forces = []

    for i in range(NumElem):
        sn    = C[i][0] - 1
        en    = C[i][1] - 1
        mid   = C[i][2] - 1
        A     = M[mid][0]
        I_val = M[mid][1]
        E_mod = M[mid][2]

        dx = XY[en][0] - XY[sn][0]
        dy = XY[en][1] - XY[sn][1]
        L  = math.sqrt(dx*dx + dy*dy)
        cx, cy = dx/L, dy/L

        # 13.1 Global end displacements
        G = E_arr[sn][:] + E_arr[en][:]
        d_global = [D[eq-1] if eq != 0 else 0.0 for eq in G]

        # 13.2 Rotation matrix
        R = element_rotation_matrix(cx, cy)

        # 13.3 Local displacements d' = R * d
        d_local = [sum(R.get(r, c) * d_global[c] for c in range(6))
                   for r in range(6)]

        # 13.4 Local stiffness
        k_local = element_local_stiffness(A, I_val, E_mod, L)

        # 13.5 Local end forces f' = k' * d'
        f_local = [sum(k_local.get(r, c) * d_local[c] for c in range(6))
                   for r in range(6)]

        all_forces.append(f_local)

    return all_forces


# ==============================================================================
# G. PRINT RESULTS
# ==============================================================================

def print_results(NumNode, NumElem, E_arr, K, F, D, all_forces):
    """
    Purpose  : Print all results — K, F, D, member end forces.
    """
    sep = "=" * 65
    print("\n" + sep)
    print("  FRAME ANALYSIS RESULTS")
    print(sep)

    # Equation numbers
    print("\n--- EQUATION NUMBER ARRAY E (0 = restrained) ---")
    print(f"  {'Node':>5}  {'UX':>6}  {'UY':>6}  {'RZ':>6}")
    for i, row in enumerate(E_arr):
        print(f"  {i+1:>5}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}")

    # Global stiffness matrix
    n = K.rows
    print(f"\n--- GLOBAL STIFFNESS MATRIX K ({n}x{n}) ---")
    for r in range(n):
        vals = "  ".join(f"{K.get(r,c):12.2f}" for c in range(n))
        print(f"  [{vals}]")

    # Load vector
    print("\n--- GLOBAL LOAD VECTOR F ---")
    for i, v in enumerate(F):
        print(f"  F[{i+1}] = {v:10.4f} kN or kN.m")

    # Displacements
    print("\n--- NODAL DISPLACEMENTS D ---")
    print(f"  {'Node':>5}  {'UX [m]':>14}  {'UY [m]':>14}  {'RZ [rad]':>14}")
    for node in range(NumNode):
        vals = [D[E_arr[node][d]-1] if E_arr[node][d] != 0 else 0.0
                for d in range(3)]
        print(f"  {node+1:>5}  {vals[0]:>14.6e}  {vals[1]:>14.6e}  {vals[2]:>14.6e}")

    # Member end forces
    print("\n--- MEMBER END FORCES (local coordinates) ---")
    print(f"  {'Elem':>5}  {'N1[kN]':>11}  {'V1[kN]':>11}  {'M1[kNm]':>11}"
          f"  {'N2[kN]':>11}  {'V2[kN]':>11}  {'M2[kNm]':>11}")
    for i, f_loc in enumerate(all_forces):
        print(f"  {i+1:>5}  " +
              "  ".join(f"{v:>11.4f}" for v in f_loc))

    print("\n" + sep)


# ==============================================================================
# MAIN DRIVER
# ==============================================================================

def run_frame_analysis():
    print("\n" + "=" * 65)
    print("  2D FRAME ANALYSIS  —  CE 4011 Assignment #2")
    print("=" * 65)

    print("\n[Phase A] Input...")
    (NumNode, NumElem, NumSupport, NumLoadJoint,
     XY, M, C, S, L) = input_phase()
    print(f"  {NumNode} nodes, {NumElem} elements, "
          f"{NumSupport} supports, {NumLoadJoint} loaded joints")

    print("\n[Phase B] Equation numbering...")
    E_arr, NumEq = equation_numbering(NumNode, S)
    print(f"  NumEq = {NumEq}")

    print("\n[Phase C] Building global stiffness K...")
    K = build_global_stiffness(NumEq, NumElem, XY, M, C, E_arr)
    print(f"  K ({NumEq}x{NumEq}) assembled.")

    print("\n[Phase D] Building load vector F...")
    F = build_load_vector(NumEq, NumLoadJoint, L, E_arr)

    print("\n[Phase E] Solving for displacements D...")
    D = solve_displacements(K, F)

    print("\n[Phase F] Member end forces...")
    forces = member_end_forces(NumElem, XY, M, C, E_arr, D)

    print_results(NumNode, NumElem, E_arr, K, F, D, forces)

    return E_arr, K, F, D, forces


if __name__ == "__main__":
    run_frame_analysis()
