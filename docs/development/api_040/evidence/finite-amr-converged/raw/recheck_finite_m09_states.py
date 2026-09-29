"""Independent NumPy recomputation from archived PoPS states; never imports pops."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(directory):
    rng = np.random.default_rng(20260928)
    g = rng.normal(size=(8, 4))
    rho = np.diag(np.tile([1., 1.1, .9, 1.2], 2))
    j = np.block([[np.zeros((4, 4)), np.eye(4)], [-np.eye(4), np.zeros((4, 4))]])
    a, b, c, k = np.eye(8) - .03*j, .03*g, -.03*.5*g.T@rho, g.T@g
    v0, p0 = rng.normal(size=8), rng.normal(size=4)
    matrix = np.block([[a, b], [c, k]])
    records = []
    for path in sorted(directory.rglob('*.npz')):
        with np.load(path, allow_pickle=False) as state:
            v, p = state['velocity'], state['potential']
            old_v = state['velocity_old'] if 'velocity_old' in state else state['old_v']
            old_p = state['potential_old'] if 'potential_old' in state else state['old_p']
        index = {'first_capture.npz': 0, 'second_capture_same_artifact.npz': 1,
                 'new_bind_after_two_refusals.npz': 2}.get(path.name, 0)
        np.testing.assert_array_equal(old_v, v0*(1+.125*index)+np.linspace(-.2,.1,8)*index)
        np.testing.assert_array_equal(old_p, p0*(1-.2*index)+np.linspace(.15,-.1,4)*index)
        rhs = np.concatenate((old_v, k@old_p))
        result = np.concatenate((v, p))
        reference = np.linalg.solve(matrix, rhs)
        residual = float(np.max(np.abs(matrix@result-rhs)))
        difference = float(np.max(np.abs(result-reference)))
        record = {'state': str(path.relative_to(directory)), 'sha256': digest(path),
                  'original_residual': residual, 'monolithic_error': difference}
        assert np.isfinite(result).all()
        assert max(residual, difference) < 1.e-11, record
        if path.name.startswith('finite-w06-'):
            end_v, end_p = 2*v-old_v, 2*p-old_p
            def energy(vel, phi):
                return .5*vel@rho@vel + phi@k@phi
            error = float(abs(energy(end_v,end_p)-energy(old_v,old_p)))
            record['cn_energy_error'] = error
            assert error < 1.e-11, record
        records.append(record)
    assert len(records) == 16, len(records)
    return {'directory': str(directory), 'state_count': len(records),
            'max_original_residual': max(r['original_residual'] for r in records),
            'max_monolithic_error': max(r['monolithic_error'] for r in records),
            'max_cn_energy_error': max(r.get('cn_energy_error',0) for r in records),
            'records': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', type=Path, required=True)
    parser.add_argument('--mpi', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema_version': 1, 'status': 'passed', 'threshold': 1.e-11,
              'source_sha256': digest(Path(__file__)), 'seed': 20260928,
              'numpy_version': np.__version__,
              'scope': 'Finite 8+4 DOF M09 witness; not a meshed PDE or global distributed solve.',
              'serial': check(args.serial), 'mpi2_rank0': check(args.mpi)}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: {k:v for k,v in row.items() if k != 'records'}
                      for name,row in report.items() if isinstance(row,dict)},indent=2))
