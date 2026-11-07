"""N3a: CPU roofline fractions of the b6 native kernels of M01 and M03, grouped by IR20 class.

Usage: roofline.py STREAM.jsonl RUN.json [RUN.json ...], with RUN.json written by face_routes.py
under the kernel timer. Kernels are recognised by the first PoPS frame of their launch stack.

Bytes follow the STREAM convention: each array element read or written once counts 8 bytes;
write-allocate traffic is not counted. The counts are those of the b6 kernels of one first-order
SSPRK2 finite-volume step (two evaluations) with recorded face speeds. Every kernel here has an
arithmetic intensity below 1.5 flop/byte, far under the ratio of the vector peak to the memory
bandwidth, so its roof is the memory bandwidth and the fraction is
(bytes / time) / (best STREAM bandwidth at the same threads).
"""
import json
import statistics
import sys

NC = {"M01": 1, "M03": 4}


def frames(name):
    return [line for line in name.split("\n")[1:] if line and line != "?"]


def kind(label, name):
    chain = frames(name)
    first = chain[0] if chain else ""
    if "memset" in label:
        return "allocation"
    if "maximum_incident_face_frequency" in first:
        return "incident_frequency"
    if "reduce_max_local" in first:
        return "frequency_reduce"
    if "copy_face_axes" in first:
        return "copy_face_axes"
    if "saxpy" in first:
        return "saxpy"
    if "fill_boundary" in first:
        return "halo_fill"
    if "maximum_face_status" in "\n".join(chain[:2]) and "reduce" in label:
        return "face_status_reduce"
    if "assemble_residual_from_face_fluxes" in "\n".join(chain[:2]):
        return "cell_status_reduce" if "reduce" in label else "residual_assembly"
    if first.startswith("void pops::runtime::program::collective_step_rejection_phase") \
            and "for_each_cell" in label:
        return "face_flux"
    if "System<" in first:
        return "system_guard_reduce" if "reduce" in label else "system_guard"
    return "other"


CLASS = {"face_flux": "face incidence", "residual_assembly": "face incidence",
         "copy_face_axes": "pointwise", "saxpy": "pointwise",
         "face_status_reduce": "axis reduction", "cell_status_reduce": "axis reduction",
         "incident_frequency": "face incidence", "frequency_reduce": "axis reduction"}


def bytes_per_step(group, case, cells, calls_per_step):
    nx, ny = cells
    n = nx*ny
    nc = NC[case]
    faces = (nx + 1)*ny + nx*(ny + 1)
    evaluations = 2  # SSPRK2
    if group == "face_flux":  # read the state once per axis; write flux, status and speed
        return evaluations*(2*n*nc + faces*(nc + 2))*8
    if group == "copy_face_axes":
        return evaluations*faces*nc*2*8
    if group == "face_status_reduce":
        return evaluations*faces*8
    if group == "residual_assembly":  # residual from faces, then copy of the candidate
        return evaluations*(faces*nc + n*(nc + 1) + 2*n*nc)*8
    if group == "cell_status_reduce":
        return evaluations*n*8
    if group == "saxpy":  # one component per call
        return calls_per_step*3*n*8
    if group == "incident_frequency":  # four face speeds per cell, one cell value written
        return evaluations*(faces + n)*8
    if group == "frequency_reduce":
        return evaluations*n*8
    return None


def stream_roof(path, threads):
    best = None
    for line in open(path):
        row = json.loads(line)
        if int(row["threads"]) == int(threads):
            value = max(row[key] for key in row if key.endswith("_GBps"))
            best = value if best is None else max(best, value)
    return best


def analyse(bench_path, stream_path):
    data = json.load(open(bench_path))
    case, cells, threads = data["case"], tuple(data["cells"]), data["threads"] or "1"
    roof = stream_roof(stream_path, threads)
    out = {"case": case, "cells": cells, "threads": int(threads), "roof_GBps": roof, "routes": {}}
    for route, kernels in data["kernels"].items():
        groups = {}
        for name, entry in kernels.items():
            label = name.split("\n")[0]
            group = kind(label, name)
            row = groups.setdefault(group, {"seconds": 0., "calls": 0, "steps": entry["steps"]})
            row["seconds"] += entry["seconds"]
            row["calls"] += entry["calls"]
        table = {}
        for group, row in groups.items():
            steps = row["steps"]
            seconds = row["seconds"]/steps
            calls = row["calls"]/steps
            moved = bytes_per_step(group, case, cells, calls)
            entry = {"s_per_step": seconds, "calls_per_step": calls, "class": CLASS.get(group)}
            if moved is not None and seconds > 0:
                bandwidth = moved/seconds/1e9
                entry.update({"GB_per_step": moved/1e9, "GBps": bandwidth,
                              "roofline_fraction": bandwidth/roof})
            table[group] = entry
        out["routes"][route] = table
    return out


if __name__ == "__main__":
    stream_path = sys.argv[1]
    results = [analyse(path, stream_path) for path in sys.argv[2:]]
    medians = {}
    for result in results:
        native = result["routes"].get("native_rusanov", {})
        for entry in native.values():
            if entry.get("class") and "roofline_fraction" in entry:
                medians.setdefault((result["threads"], entry["class"]), []).append(
                    entry["roofline_fraction"])
    summary = {"%d threads, %s" % key: {"median": statistics.median(values), "n": len(values),
                                         "values": values}
               for key, values in sorted(medians.items())}
    print(json.dumps({"results": results, "native_class_medians": summary}, indent=1))
