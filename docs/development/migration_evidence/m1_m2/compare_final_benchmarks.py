"""Compare frozen migration benchmark records without inventing regression thresholds."""
import argparse, hashlib, json
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report={"schema":"pops.migration.benchmark-comparison.v1","comparisons":[],"files":[],"limitations":["Host timings are observations, not a zero-overhead or speedup guarantee.","M0 benchmark source included the recorded explicit ExecutionLane API repair.","No GPU hardware, hardware memory-traffic counters, or aggregate MPI-worker peak RSS was measured.","No comparable baseline native shared-kernel witness: baseline authoring failed before native execution."]}
    for suffix in ("", "-mpi2"):
        before=args.root/("baseline-benchmarks"+suffix+".json")
        after=args.root/(args.candidate+suffix+".json")
        for path in (before,after):
            report["files"].append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        a,b=rows(before),rows(after)
        assert len(a)==len(b)==4
        for x,y in zip(a,b):
            key=(x["record_type"],x["case"],x["variant"])
            assert key==(y["record_type"],y["case"],y["variant"])
            assert x["parameters"]==y["parameters"],key
            assert x["validation"]["passed"] and y["validation"]["passed"],key
            for field in ("compiler","build_type","execution_space","execution_concurrency","mpi_ranks","real_bytes"):
                assert x["metadata"][field]==y["metadata"][field],(key,field)
            if key[0]!="measurement":continue
            tx,ty=x["timing"],y["timing"]
            assert {k:v for k,v in tx.items() if k!="statistics"}=={k:v for k,v in ty.items() if k!="statistics"},key
            sx,sy=tx["statistics"],ty["statistics"]
            report["comparisons"].append({"case":key[1],"variant":key[2],"ranks":y["metadata"]["mpi_ranks"],"baseline_revision":x["metadata"]["git_sha"],"candidate_revision":y["metadata"]["git_sha"],"candidate_dirty":y["metadata"]["source_dirty"],"baseline_median_seconds":sx["median"],"candidate_median_seconds":sy["median"],"candidate_over_baseline":sy["median"]/sx["median"],"baseline_mad_seconds":sx["mad"],"candidate_mad_seconds":sy["mad"],"sample_count":sy["count"],"validation":y["validation"]})
    report["valid_records"]=8
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"valid_records":8,"comparisons":len(report["comparisons"])}))

if __name__=="__main__":main()
