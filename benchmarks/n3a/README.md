# N3a face-route measurements (ROM-82)

This campaign measures the N3a pilot on CPU: the Rusanov and HLL faces written in Python in
`pops.lib.numerics` against b6's native `riemann.Rusanov()` and `riemann.HLL()` on M01 and M03.
It is not routine CI. It records times; it asserts no time threshold.

- `face_routes.py`: compiles each route from the same builder, binds the same initial data, warms
  up, then advances the routes in interleaved `A B B A` blocks of fixed accepted steps. It reports
  the median time per step, the throughput ratio against the first route (PF10 (1), OD9: 0.9),
  and whether the final states are bitwise equal, with their L1 and L-infinity differences.
  The `python_hll` route is the derived `hll(F, Uh)`, whose face reads the signed wave speeds
  that the model declares (ROM-99); like `native_hll`, it needs `--signed`.
- `kernel_timer.cpp`: a Kokkos Tools library for host backends. Kernels are keyed by label and
  launch stack, so the many `pops_for_each_cell` kernels are told apart.
- `stream.cpp`: copy, scale, add and triad bandwidth with the same OpenMP threads.
- `roofline.py`: groups the kernels of each route by IR20 class and divides their effective
  bandwidth by the best STREAM bandwidth (OD24). Its byte counts follow the b6 kernels of a
  first-order SSPRK2 finite-volume step with recorded face speeds; they are stated in the script.

## Run

From the repository root, with PoPS installed for one process (serial communicator):

```sh
mkdir -p build/n3a
c++ -O2 -fPIC -shared benchmarks/n3a/kernel_timer.cpp -o build/n3a/libn3a_kernel_timer.so -ldl
c++ -O3 -march=native -fopenmp benchmarks/n3a/stream.cpp -o build/n3a/stream
export PYTHONPATH=$PWD KOKKOS_TOOLS_LIBS=$PWD/build/n3a/libn3a_kernel_timer.so
OMP_NUM_THREADS=1 python benchmarks/n3a/face_routes.py M01 4096x4096 --steps 3 --out m01.json
OMP_NUM_THREADS=1 python benchmarks/n3a/face_routes.py M03 2048x2048 --steps 2 --out m03.json
OMP_NUM_THREADS=1 python benchmarks/n3a/face_routes.py M01 4096x4096 --signed \
    --routes native_hll,python_hll --steps 3 --out m01_hll.json
OMP_NUM_THREADS=1 python benchmarks/n3a/face_routes.py M03 2048x2048 --signed \
    --routes native_hll,python_hll --steps 2 --out m03_hll.json
OMP_NUM_THREADS=1 build/n3a/stream 200000000 10 >> stream.jsonl
python benchmarks/n3a/roofline.py stream.jsonl m01.json m03.json
```

Run one measurement at a time: a 4096² M01 pair holds about 13 GB, and a concurrent build or
test perturbs the samples. With OpenMP threads, also set `OMP_PROC_BIND=spread` and
`OMP_PLACES=threads`.
