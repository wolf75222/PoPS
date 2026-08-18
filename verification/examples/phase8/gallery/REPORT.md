# Verification report

suite: pr
repository: wolf75222/PoPS
repository_sha: 0123456789abcdef0123456789abcdef01234567
max_nodes: 2

## 1. Components covered
euler

## 2. Backends and dimensions tested
backends: KokkosSerial
dimensions: 1

## 3. Failures
none

## 4. Orders measured
CP-02 spatial density observed_order=1.95 threshold=1.8

## 5. AMR order and invariants
AMR not run in local pr

## 6. Poisson and the field
Poisson campaign not run in local pr

## 7. Coupling phase, sign, and energy
coupling not run in local pr

## 8. Parallel invariance (ranks, threads, GPU)
parallel invariance not run in local pr

## 9. Performance on one and two nodes
performance not measured in local pr

## 10. Not tested
AMR not run in local pr
Poisson campaign not run in local pr
coupling not run in local pr
parallel invariance not run in local pr
performance not measured in local pr

## Visual gallery

- `orders_heatmap`: `analysis/figures/publication/orders_heatmap.svg`
- `amr_degradation`: `analysis/figures/publication/amr_degradation.svg`
- `component_coverage`: `analysis/figures/publication/component_coverage.svg`
- `backend_parity`: `analysis/figures/publication/backend_parity.svg`
- `conservation_dashboard`: `analysis/figures/publication/conservation_dashboard.svg`
- `poisson_dashboard`: `analysis/figures/publication/poisson_dashboard.svg`
- `temporal_dashboard`: `analysis/figures/publication/temporal_dashboard.svg`
- `amr_dashboard`: `analysis/figures/publication/amr_dashboard.svg`
- `performance_dashboard`: `analysis/figures/publication/performance_dashboard.svg`
- `failure_map`: `analysis/figures/publication/failure_map.svg`
