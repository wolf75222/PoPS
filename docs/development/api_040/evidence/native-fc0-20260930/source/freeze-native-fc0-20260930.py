from pathlib import Path
import json,hashlib,shutil,xml.etree.ElementTree as ET
ws=Path("/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops")
source=ws/"outputs"
target=Path("/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS")/"docs/development/api_040/evidence/native-fc0-20260930"
if target.exists(): raise SystemExit("Archive exists; refusing overwrite")
target.mkdir(parents=True)
groups=[
 "installed-coherent-units-fc0-20260930",
 "installed-coherent-units-fc0-corrected-20260930",
 "installed-m26-m27-native-dim1-fc0-20260930",
 "installed-m27-native-dim1-restart128-fc0-20260930",
 "installed-m26-m27-mpi2-fc0-20260930",
 "installed-integral-nd-native-dim2-fc0-20260930",
 "installed-integral-native-initial-checkpoint-fc0-20260930",
 "installed-initial-checkpoint-controls-fc0-20260930",
 "installed-coupled-gradient-dim2-mpi2-fc0-20260930",
 "native-integral-coverage-cpp-20260930",
]
allowed={".json",".npz",".xml",".log",".txt",".cpp",".hpp",".inc",".py"}
files=[];group_receipts=[];states=[]
def retain(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(src,dst)
 content=dst.read_bytes()
 files.append({"path":str(dst.relative_to(target)),"origin":str(src),"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()})
for group in groups:
 root=source/group
 if not root.is_dir(): raise SystemExit("missing source group:"+group)
 prefix=Path("runs")/group
 counts={}
 for src in sorted(root.rglob("*")):
  if not src.is_file() or src.is_symlink() or src.suffix not in allowed:continue
  rel=src.relative_to(root)
  if any(part in ("pops-native-cache","__pycache__", ".pytest_cache") for part in rel.parts):continue
  if any(part in ("pytest-tmp","rank0-tmp","rank1-tmp") for part in rel.parts) and not any(label in str(rel) for label in ("test_finite_interaction_two_de","test_mixed_linear_three_grids","test_dim2_fourier_faces_and_co")):continue
  retain(src,target/prefix/rel)
  if src.suffix==".xml":
   cases=list(ET.parse(src).iter("testcase"))
   counts[str(rel)]={"total":len(cases),"failed":sum(c.find("failure") is not None or c.find("error") is not None for c in cases),"skipped":sum(c.find("skipped") is not None for c in cases)}
  state_path=str(prefix/rel)
  if src.suffix==".npz" and "test_finite_interaction_two_de" in str(rel) and src.name.startswith("state_"):
   states.append({"kind":"m26_finite12_v1","path":state_path})
  if src.suffix==".npz" and "test_mixed_linear_three_grids" in str(rel) and src.name.startswith("state_") and group in ("installed-m27-native-dim1-restart128-fc0-20260930","installed-m26-m27-mpi2-fc0-20260930"):
   states.append({"kind":"m27_mixed_linear_v1","path":state_path})
  if src.name=="coupled_gradient_dim2.npz":
   ledger=str(prefix/rel.with_name("coupled_gradient_dim2_ledger.json"))
   states.append({"kind":"coupled_gradient_dim2_v1","path":state_path,"ledger":ledger})
 group_receipts.append({"name":group,"junit":counts,"outer_result":str(prefix/"result.json") if (root/"result.json").is_file() else None})
extra=[
 "coverage-preflight-diffusion-mpi3-20260930.log",
 "coverage-preflight-diffusion-mpi3-tcp-20260930.log",
 "coverage-preflight-diffusion-mpi3-tcp-20260930.xml",
 "integral-transaction-native-tcp-20260930.log",
 "integral-transaction-native-tcp-20260930.xml",
 "mpi-provider-tcp-diffusion-probe-20260930.log",
 "build-coverage-preflight-native-cpp-20260930.log",
 "build-initial-checkpoint-python-refresh-fc0-20260930.log",
 "m27-native-saved-independent-recompute-20260930.json",
]
for name in extra:
 p=source/name
 if not p.is_file():raise SystemExit("missing extra:"+name)
 retain(p,target/"supplemental"/name)
recipe=ws/"outputs/freeze-native-fc0-20260930.py"
if recipe.is_file():retain(recipe,target/"source/freeze-native-fc0-20260930.py")
manifest={"schema_version":1,"native_scope":{"sdk_sha256":"fc0bfd9a0ab1a6036a2e941dd16fdcbfc486125d31b8679d465d68111d16e959","native_abi_version":4,"dimension1_native_sha256":"835759c416fd40ebfe1a61ecd466feb07580c798ad12fb2488101268f2b65941","dimension2_native_sha256":"a280d9ef8d95c261be61b61f6d0b1f8da0be7a24515380eaaa03cfd9489f3943","local_backend":"Apple LLVM21, Kokkos5.2.0 CPU, MPICH4.1.2 TCP, one thread","mpi_receipts":"M26/M27 MPI2 before/after installed identity; C++ transaction MPI2 and diffusion preflight MPI3"},"limitations":["Archived receipts qualify their recorded ABI4 source and native images, not later ABI5 builds.","Original serial reception authenticates installed PoPS before execution; it does not claim an after-execution authentication.","M26 is a twelve-DOF measured interaction at one actual cell, not an aggregation PDE.","M27 is the frozen linear mixed pair, not nonlinear Cahn-Hilliard.","ND2 has two passed native cases inside an outer run whose six Integral checkpoint cases failed.","Both initial Integral checkpoint failure stages are retained: undeclared controller, then missing run provenance.","A source-only package-path assertion accounts for six initial-controls fixture failures; 116 other cases passed.","Default OFI finalization failure and explicit TCP reception are both retained.","The later ND2 MPI launch was refused by source/SDK authentication before any tests after new headers were integrated.","No Python Dim3, GPU, ROMEO or official OpenMPI matrix reception is asserted."],"groups":group_receipts,"scientific_states":states,"files":sorted(files,key=lambda row:row["path"])}
(target/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
print(json.dumps({"archive":str(target),"payloads":len(files),"scientific_states":len(states),"bytes":sum(row["bytes"] for row in files),"groups":len(group_receipts)},indent=2))

