# N3 measurements

`rom107_adaptive_faces.py` (ROM-107) runs the native and the Python Rusanov face of M01 and M03
under `AdaptiveCFL`, with the guard of the step strategy's face frequency rule or, with
`--rule incident_sum`, with the guard of the base. It reports accepted steps, final clocks, the
reason of a refused attempt and the final states against the native face. It is not routine CI and
asserts nothing; the core ledger entry `face-frequency-rule-of-the-step-strategy` records its
output.

```sh
export PYTHONPATH=$PWD
python benchmarks/n3/rom107_adaptive_faces.py M03 --out m03.json
python benchmarks/n3/rom107_adaptive_faces.py M03 --rule incident_sum --out m03_base.json
```
