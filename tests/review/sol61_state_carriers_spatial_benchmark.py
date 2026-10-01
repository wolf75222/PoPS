"""Reproduce the bounded STD host comparison; never builds a native runtime TU."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")
args.directory.mkdir(parents=True, exist_ok=True)
old = subprocess.check_output(["git", "show", "86a71b6:include/pops/runtime/checkpoint/state_carriers.hpp"], cwd=root, text=True)
old = old.replace("namespace pops::runtime::checkpoint {", "namespace pops::runtime::checkpoint_old {")
old = old.replace("/// A global archive covers every block", "inline std::uint64_t measured_pair_checks = 0;\n\n/// A global archive covers every block")
old = old.replace("for (std::size_t j = begin; block == 0 && j + 1 < at; ++j) {", "for (std::size_t j = begin; block == 0 && j + 1 < at; ++j) {\n          ++measured_pair_checks;")
header = args.directory / "old_instrumented.hpp"
header.write_text(old)
binary = args.directory / "spatial_host"
command = ["c++", "-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror", '-DOLD_HEADER="'+str(header)+'"', "-I"+str(root/"include"), str(root/"tests/review/sol61_state_carriers_spatial_host.cpp"), "-o", str(binary)]
subprocess.run(command, check=True)
report = {"status": "SOURCE_ONLY", "baseline_commit": "86a71b6", "compiler": subprocess.check_output(["c++", "--version"], text=True), "platform": platform.platform(), "command": command, "threads": 1, "binary_bytes": binary.stat().st_size, "runs": []}
report["baseline_commit"] = subprocess.check_output(["git", "rev-parse", "86a71b6"], cwd=root, text=True).strip()
report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (header, root/"include/pops/runtime/checkpoint/state_carriers.hpp", root/"tests/review/sol61_state_carriers_spatial_host.cpp")}
report["combined_binary_sections"] = subprocess.check_output(["size", "-m", str(binary)], text=True) if platform.system() == "Darwin" else subprocess.check_output(["size", str(binary)], text=True)
for run in range(args.runs):
    result = subprocess.run((["/usr/bin/time", "-l"] if platform.system() == "Darwin" else ["/usr/bin/time", "-v"])+[str(binary)], check=True, capture_output=True, text=True)
    (args.directory / ("run-%d.csv" % run)).write_text(result.stdout)
    (args.directory / ("run-%d.resources.txt" % run)).write_text(result.stderr)
    report["runs"].append({"csv": result.stdout, "resources": result.stderr})
    print("completed host comparison run", run, flush=True)
# A single-source decode+validate probe gives comparable executable section sizes for each
# implementation. It is a host cost witness, not the installed DSO or native runtime size.
for label, include, namespace in (("old", str(header), "pops::runtime::checkpoint_old"), ("new", "pops/runtime/checkpoint/state_carriers.hpp", "pops::runtime::checkpoint")):
    source = args.directory / (label+"-size.cpp")
    source.write_text('#include <'+include+'>\n#include <fstream>\n#include <iterator>\nint main(int n,char** v){if(n!=2)return 2;std::ifstream f(v[1],std::ios::binary);std::vector<std::uint8_t>b((std::istreambuf_iterator<char>(f)),{});auto a='+namespace+'::decode_state_carriers<2>(b);'+namespace+'::validate_complete_state_carriers(a);return 0;}\n')
    exe = args.directory / (label+"-size")
    subprocess.run(["c++", "-std=c++20", "-O3", "-I"+str(root/"include"), str(source), "-o", str(exe)], check=True)
    report[label+"_probe"] = {"binary_bytes": exe.stat().st_size, "sections": subprocess.check_output(["size", "-m", str(exe)], text=True) if platform.system() == "Darwin" else subprocess.check_output(["size", str(exe)], text=True)}
(args.directory / "report.json").write_text(json.dumps(report, indent=2)+"\n")
print(args.directory / "report.json")
