"""Persistence of real Native producer witnesses; never constructs a synthetic image."""
import hashlib
import json
from pathlib import Path

SCHEMA="pops.amr.field-candidate-observation@1"

def save_rank_observations(directory,phase,rank,rows):
    """Save rank-local immutable full image bytes and exact returned metadata."""
    if not isinstance(rows,list) or not rows:
        raise ValueError("missing executed Field candidate observations")
    if type(rank) is not int or rank<0 or phase not in ("initial","accepted"):
        raise ValueError("invalid Field observation persistence authority")
    result=[];keys=set()
    for index,row in enumerate(rows):
        if not isinstance(row,dict) or row.get("schema")!=SCHEMA or row.get("accepted_publication") is not False:
            raise ValueError("invalid Field candidate observation contract")
        key=(row["provider_slot"],row["consumer_block"],row["consumer_level"])
        if key in keys:raise ValueError("duplicate Field consumer invocation")
        keys.add(key)
        payload=row["carrier_bytes"]
        if type(payload) is not bytes or not payload.startswith(b"POPSCAR1"):
            raise ValueError("missing authentic full Field carrier bytes")
        path=Path(directory)/f"{phase}-field-candidate-rank{rank}-{index}.bin"
        path.write_bytes(payload)
        metadata={key:value for key,value in row.items() if key!="carrier_bytes"}
        metadata["carrier"]={"path":str(path.resolve()),"sha256":hashlib.sha256(payload).hexdigest(),"size_bytes":len(payload)}
        result.append(metadata)
    target=Path(directory)/f"{phase}-field-candidate-rank{rank}.json"
    target.write_text(json.dumps({"schema":SCHEMA,"rank":rank,"observations":result},indent=2)+"\n")
    return target
