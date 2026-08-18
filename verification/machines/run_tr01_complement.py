#!/usr/bin/env python3
"""Run one exact-rank TR-01 configuration. Does not invent order from smoke."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _installed_native_root() -> Path | None:
    try:
        import pops
    except ImportError:
        return None
    for item in getattr(pops, "__path__", ()):
        candidate = Path(item).absolute() / "_native"
        if (candidate / "variants.json").is_file():
            return candidate
    return None


_INSTALLED_NATIVE = _installed_native_root()
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))
if _INSTALLED_NATIVE is not None and not os.environ.get("POPS_NATIVE_VARIANTS_ROOT"):
    os.environ["POPS_NATIVE_VARIANTS_ROOT"] = str(_INSTALLED_NATIVE)

RUN = ROOT / "verification" / "cases" / "transport" / "advection_sine" / "run.py"
ANALYZE = ROOT / "verification" / "cases" / "transport" / "advection_sine" / "analyze.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build" / "verification" / "TR-01",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config-id", default=None)
    parser.add_argument(
        "--resolutions",
        default=None,
        help="comma-separated n, default 16 or 16,32,64,128",
    )
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument(
        "--dts",
        default=None,
        help="comma-separated dt; default CFL≤0.45 (dt≤0.005 at N=64)",
    )
    parser.add_argument("--n-cells", type=int, default=64)
    args = parser.parse_args()
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    if args.temporal:
        run = _load(RUN, "tr01_run")
        if args.dts:
            dts = tuple(float(item) for item in args.dts.split(",") if item)
        else:
            dts = run.default_temporal_dts(n_cells=int(args.n_cells))
        request = CampaignRequest(
            case_id="TR-01",
            pops_native_dim=args.dim,
            suite="pr",
            execution_space=os.environ.get("POPS_TR01_SPACE", "KokkosSerial"),
            mpi_mode=os.environ.get("POPS_TR01_MPI", "off"),
            min_resolution=int(args.n_cells),
            resources=CampaignResources(
                nodes=1,
                mpi_ranks=int(os.environ.get("SLURM_NTASKS", "1")),
                omp_threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
                resolutions=(int(args.n_cells),),
            ),
            evidence_status="required",
            output_dir=Path(args.out) / f"dim{args.dim}",
        )
        analyze = _load(ANALYZE, "tr01_analyze")
        output = Path(request.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        campaign = run.run_temporal_campaign(
            dts,
            n_cells=int(args.n_cells),
            request=request,
            config_id=args.config_id,
            output_dir=output,
        )
        analyze.write_native_campaign_report(output, campaign)
        claim = analyze.evaluate_order_claim(campaign)
        print(
            f"temporal dim={args.dim} n={args.n_cells} dts={list(dts)} "
            f"verdict={claim['verdict']} orders={claim['orders']}",
            flush=True,
        )
        return 0 if claim["order_pass"] else 1
    if args.resolutions:
        resolutions = tuple(int(item) for item in args.resolutions.split(",") if item)
    elif args.smoke:
        resolutions = (16,)
    else:
        resolutions = (16, 32, 64, 128)
    request = CampaignRequest(
        case_id="TR-01",
        pops_native_dim=args.dim,
        suite="pr",
        execution_space=os.environ.get("POPS_TR01_SPACE", "KokkosSerial"),
        mpi_mode=os.environ.get("POPS_TR01_MPI", "off"),
        min_resolution=resolutions[0],
        resources=CampaignResources(
            nodes=1,
            mpi_ranks=int(os.environ.get("SLURM_NTASKS", "1")),
            omp_threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
            resolutions=resolutions,
        ),
        evidence_status="required",
        output_dir=Path(args.out) / f"dim{args.dim}",
    )
    run = _load(RUN, "tr01_run")
    analyze = _load(ANALYZE, "tr01_analyze")
    output = Path(request.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if len(resolutions) < 4:
        fields = {}
        oracles = {}
        volumes = {}
        linfs = []
        label = None
        for n_cells in resolutions:
            payload = run.run_native(
                n_cells,
                request=request,
                config_id=args.config_id,
                output_dir=output,
            )
            fields[n_cells] = payload["field"]
            oracles[n_cells] = payload["oracle"]
            volumes[n_cells] = payload["volumes"]
            linfs.append(payload["diagnostics"]["linf"])
            label = payload.get("label")
        claim = analyze.evaluate_order_claim(
            {
                "source": "native",
                "family": "global",
                "dt_scaling": "cfl",
                "reconstruction": "weno5z",
                "resolutions": resolutions,
                "fields": fields,
                "oracles": oracles,
                "volumes": volumes,
            }
        )
        (output / "smoke.json").write_text(
            json.dumps(
                {
                    "dim": args.dim,
                    "label": label,
                    "verdict": claim["verdict"],
                    "order_pass": claim["order_pass"],
                    "resolutions": list(resolutions),
                    "linf": linfs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"partial dim={args.dim} label={label} "
            f"verdict={claim['verdict']} n={list(resolutions)} linf={linfs}",
            flush=True,
        )
        return 0
    campaign = run.run_order_campaign(
        resolutions, request=request, config_id=args.config_id, output_dir=output
    )
    analyze.write_native_campaign_report(output, campaign)
    claim = analyze.evaluate_order_claim(campaign)
    print(
        f"series dim={args.dim} label={campaign.get('label')} "
        f"verdict={claim['verdict']} orders={claim['orders']}",
        flush=True,
    )
    return 0 if claim["order_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
