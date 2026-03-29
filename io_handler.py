"""
================================================================================
Module      : io_handler.py
Purpose     : Reads structural model input from a JSON file and writes
              analysis results (displacements, reactions, element forces)
              to a formatted text report.
Input format: JSON file (see INPUT FORMAT section below)
Output      : Console + text file report
Assumptions : 2D problems only; truss or frame elements
Units       : kN, m, kN·m
Author      : CE 4011 Assignment #2
================================================================================

INPUT FORMAT (JSON):
{
  "type": "truss" | "frame",
  "nodes": [
    {"id": 1, "x": 0.0, "y": 0.0},
    ...
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "E": 200000, "A": 0.01},      // truss
    {"id": 1, "node_i": 1, "node_j": 2, "E": 200000, "A": 0.01, "I": 1e-4},  // frame
    ...
  ],
  "supports": [
    {"node": 1, "ux": true, "uy": true},           // truss
    {"node": 1, "ux": true, "uy": true, "rz": true}, // frame
    ...
  ],
  "loads": [
    {"node": 2, "Fx": 100.0, "Fy": -50.0},
    ...
  ]
}
"""

import json


def read_input(filepath: str) -> dict:
    """
    Purpose : Read and parse a JSON model input file.
    Inputs  : filepath – path to .json file (str)
    Outputs : dict with keys: type, nodes, elements, supports, loads
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    _validate_input(data)
    return data


def _validate_input(data: dict):
    """
    Purpose : Basic validation of input dict.
    Raises  : ValueError on missing required fields.
    """
    for key in ['type', 'nodes', 'elements', 'supports', 'loads']:
        if key not in data:
            raise ValueError(f"Input file missing required key: '{key}'")
    if data['type'] not in ('truss', 'frame'):
        raise ValueError("'type' must be 'truss' or 'frame'.")


def write_results(filepath: str, model_type: str, nodes, elements,
                  displacements: list, reactions: dict,
                  element_forces: list, restrained_dofs: list):
    """
    Purpose : Write analysis results to a formatted text file.
    Inputs  : filepath       – output file path (str)
              model_type     – 'truss' or 'frame' (str)
              nodes          – list of Node objects
              elements       – list of element objects
              displacements  – global displacement vector (list of floats)
              reactions      – dict {dof_id: force_value}
              element_forces – list of per-element force lists
              restrained_dofs – list of restrained DOF indices
    Outputs : text file at filepath
    """
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("  STRUCTURAL ANALYSIS RESULTS")
    lines.append(f"  Model type: {model_type.upper()}")
    lines.append(sep)

    # Nodal displacements
    lines.append("\n--- NODAL DISPLACEMENTS ---")
    if model_type == 'truss':
        lines.append(f"{'Node':>6}  {'UX [m]':>14}  {'UY [m]':>14}")
        for node in nodes:
            ux = displacements[node.dof_ids[0]]
            uy = displacements[node.dof_ids[1]]
            lines.append(f"{node.node_id:>6}  {ux:>14.6e}  {uy:>14.6e}")
    else:
        lines.append(f"{'Node':>6}  {'UX [m]':>14}  {'UY [m]':>14}  {'RZ [rad]':>14}")
        for node in nodes:
            ux = displacements[node.dof_ids[0]]
            uy = displacements[node.dof_ids[1]]
            rz = displacements[node.dof_ids[2]]
            lines.append(f"{node.node_id:>6}  {ux:>14.6e}  {uy:>14.6e}  {rz:>14.6e}")

    # Support reactions
    lines.append("\n--- SUPPORT REACTIONS ---")
    dof_label = {0: 'UX', 1: 'UY', 2: 'RZ'}
    lines.append(f"{'DOF':>8}  {'Reaction [kN or kN.m]':>22}")
    for dof, rxn in reactions.items():
        node_id = dof // (3 if model_type == 'frame' else 2) + 1
        local_d = dof % (3 if model_type == 'frame' else 2)
        label = dof_label.get(local_d, str(local_d))
        lines.append(f"  Node {node_id} {label:>3}  {rxn:>22.6e}")

    # Element forces
    lines.append("\n--- ELEMENT FORCES ---")
    if model_type == 'truss':
        lines.append(f"{'Elem':>6}  {'Axial Force [kN]':>18}  {'Status':>10}")
        for i, elem in enumerate(elements):
            N = element_forces[i]
            status = "TENSION" if N >= 0 else "COMPRESSION"
            lines.append(f"{elem.elem_id:>6}  {N:>18.6e}  {status:>10}")
    else:
        lines.append(f"{'Elem':>6}  {'N_i [kN]':>12}  {'V_i [kN]':>12}  "
                     f"{'M_i [kN.m]':>12}  {'N_j [kN]':>12}  {'V_j [kN]':>12}  {'M_j [kN.m]':>12}")
        for i, elem in enumerate(elements):
            f_loc = element_forces[i]
            lines.append(
                f"{elem.elem_id:>6}  "
                f"{f_loc[0]:>12.4e}  {f_loc[1]:>12.4e}  {f_loc[2]:>12.4e}  "
                f"{f_loc[3]:>12.4e}  {f_loc[4]:>12.4e}  {f_loc[5]:>12.4e}"
            )

    lines.append("\n" + sep)
    result_text = "\n".join(lines)

    print(result_text)
    with open(filepath, 'w') as f:
        f.write(result_text + "\n")
    print(f"\n[Results saved to: {filepath}]")
