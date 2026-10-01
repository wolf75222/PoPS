"""Exact mathematical counterexamples for a proposed D(q) realization.

This is a Fraction face-law oracle, not a solver or a native runtime emulator.
No PoPS, NumPy, DSO, prototype, build or JIT is imported/executed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json


def coefficients(state, base, captures):
    return tuple(tuple(tuple(entry*(1+captures[cell]+sum(component[cell]**2 for component in state))
                             for cell in range(len(captures))) for entry in row) for row in base)


def face_action(state, matrices, spacing):
    """Periodic -div_h(D_face grad_h q); each face shared by its two cells."""
    width, cells = len(state), len(state[0])
    faces = tuple(tuple(-sum((matrices[row][column][cell]+matrices[row][column][(cell+1)%cells])/2
                             *(state[column][(cell+1)%cells]-state[column][cell])/spacing
                             for column in range(width)) for cell in range(cells)) for row in range(width))
    return tuple(tuple((faces[row][cell]-faces[row][(cell-1)%cells])/spacing
                       for cell in range(cells)) for row in range(width))


def tangent(state, direction, base, captures, spacing):
    matrices = coefficients(state, base, captures)
    delta = tuple(tuple(tuple(entry*2*sum(state[c][cell]*direction[c][cell] for c in range(len(state)))
                              for cell in range(len(captures))) for entry in row) for row in base)
    frozen_term = face_action(direction, matrices, spacing)
    coefficient_term = face_action(state, delta, spacing)
    return tuple(tuple(a+b for a,b in zip(left,right)) for left,right in zip(frozen_term,coefficient_term))


def shifted(state, direction, amount):
    return tuple(tuple(q+amount*v for q,v in zip(row,vector)) for row,vector in zip(state,direction))


def finite_difference(state, direction, base, captures, spacing, amount):
    plus, minus = shifted(state,direction,amount), shifted(state,direction,-amount)
    p = face_action(plus, coefficients(plus,base,captures), spacing)
    m = face_action(minus, coefficients(minus,base,captures), spacing)
    return tuple(tuple((a-b)/(2*amount) for a,b in zip(left,right)) for left,right in zip(p,m))


def run():
    state, direction, base, capture, spacing = ((Q(1),Q(2)),), ((Q(1),Q(0)),), ((Q(1),),), (Q(0),Q(0)), Q(1,2)
    original = face_action(state, coefficients(state,base,capture), spacing)
    zero = ((Q(0),Q(0)),)
    frozen_zero = face_action(state, coefficients(zero,base,capture), spacing)
    derivative = tangent(state,direction,base,capture,spacing)
    frozen_derivative = face_action(direction, coefficients(state,base,capture), spacing)
    assert original == ((Q(-28),Q(28)),)
    assert frozen_zero == ((Q(-8),Q(8)),)
    assert derivative == ((Q(20),Q(-20)),)
    assert frozen_derivative == ((Q(28),Q(-28)),)
    for amount in (Q(1,4),Q(1,8),Q(1,16)):
        fd = finite_difference(state,direction,base,capture,spacing,amount)
        assert fd[0][0] == 20+4*amount**2 and fd[0][1] == -20-4*amount**2

    # Restricting a nonlinear constitutive map and evaluating it do not commute.
    mean_q = (Q(1)+Q(3))/2
    evaluate_after_restriction = 1+mean_q**2
    restrict_after_evaluation = ((1+Q(1)**2)+(1+Q(3)**2))/2
    assert (evaluate_after_restriction,restrict_after_evaluation) == (Q(5),Q(6))

    properties = []
    for width,cells in ((3,5),(5,7)):
        q = tuple(tuple(Q((row+1)*(cell+2),11) for cell in range(cells)) for row in range(width))
        v = tuple(tuple(Q((row-cell)%5-2,13) for cell in range(cells)) for row in range(width))
        alpha = tuple(Q(cell-2,17) for cell in range(cells))
        matrix = tuple(tuple(Q((2 if i==j else (-1 if i<j else 1)),i+j+3)
                             for j in range(width)) for i in range(width))
        h = Q(1,cells)
        full = face_action(q,coefficients(q,matrix,alpha),h)
        exact = tangent(q,v,matrix,alpha,h)
        assert all(sum(h*entry for entry in row)==0 for row in full)
        a,b = Q(1,8),Q(1,16)
        fd_a,fd_b = (finite_difference(q,v,matrix,alpha,h,step) for step in (a,b))
        assert all(x-e == 4*(y-e) for row_x,row_y,row_e in zip(fd_a,fd_b,exact)
                   for x,y,e in zip(row_x,row_y,row_e))
        order = tuple(reversed(range(width)))
        permuted_matrix = tuple(tuple(matrix[i][j] for j in order) for i in order)
        permuted = face_action(tuple(q[i] for i in order),
            coefficients(tuple(q[i] for i in order),permuted_matrix,alpha),h)
        assert permuted == tuple(full[i] for i in order)
        frozen = face_action(v,coefficients(q,matrix,alpha),h)
        assert frozen != exact
        properties.append({"components":width,"cells":cells,"conservative":True,
                           "permutation_exact":True,"central_full_F_quadratic_error":True,
                           "frozen_D_derivative_is_different":True})
    return {"kind":"exact-math-candidate-diffusion-design-counterexample","native_execution":False,
            "runtime_emulation":False,"source_baseline":"b39f4a9906f904cd2f857c2874878da8aa84084f",
            "scalar":{"q":["1","2"],"direction":["1","0"],"spacing":"1/2",
                      "D":"1+q^2","original_F":["-28","28"],"frozen_at_zero_F":["-8","8"],
                      "original_DF":["20","-20"],"frozen_at_q_DF":["28","-28"],
                      "central_DF_first_component":"20+4*epsilon^2"},
            "restriction":{"D(mean_q)":"5","mean_D(q)":"6"},"vector_properties":properties}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="fresh external JSON path; existing files are refused")
    args = parser.parse_args()
    payload = json.dumps(run(),indent=2,sort_keys=True)+"\n"
    if args.output:
        with open(args.output,"x",encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload,end="")
