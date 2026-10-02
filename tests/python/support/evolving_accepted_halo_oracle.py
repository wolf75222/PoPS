"""Independent full grown-carrier oracle for the declared affine source fixture."""
import numpy as np

HALO_ROWS=[["pops.amr.accepted-halo-preparation@1","candidate_accepted_clock","all_state_components","1","1"]]

def halo_rows(rows):
    assert type(rows) is list and len(rows)==1 and type(rows[0]) is list
    assert all(type(v) is str for v in rows[0])
    assert rows==HALO_ROWS, rows

def affine_bound(scale,elapsed,steps,dim=2):
    # Native MC slope: 6 FP arithmetic ops + offset multiply/divide/add=3 peraxis.
    # Two macrosteps, at most3 child substeps+1 parent; each has affineupdate2,
    # and anchor restriction:4 differences+4 sums+division+anchoradd=10. Initial reconstruction+two comparison subtractions.
    per_fill=dim*(6+3)
    k=steps*(3+1)*(per_fill+2+10)+per_fill+2
    eps=np.finfo(np.float64).eps;gamma=k*eps/(1-k*eps)
    # MC at ratio2 has offset≤1/4; its minmod slopes are4-Lipschitz.
    # Perturbation amplification ≤1+dim for this one coarse→fine interface.
    return gamma*(1+dim)*(scale+abs(elapsed)+1)

def full_carrier(initial,current,geometry,elapsed,steps,constant):
    for name,value in (("dim",2),("real",64),("shard",-1),("levels",2),("blocks",["marker"])):
        assert initial[name]==current[name]==value,(name,initial[name],current[name])
    assert initial["ranks"]==current["ranks"]
    expected={}
    assert len(geometry)==2
    for level,selected in enumerate(geometry):
        assert selected
        for i,row in enumerate(selected):
            assert len(row)==4 and all(type(v) is int for v in row)
            ylo,xlo,yend,xend=row
            assert xlo<xend and ylo<yend
            expected[(0,level,i)]=[(xlo,xend-1),(ylo,yend-1)]
    left={p["key"]:p for p in initial["patches"]};right={p["key"]:p for p in current["patches"]}
    assert len(left)==len(initial["patches"]) and len(right)==len(current["patches"])
    assert left.keys()==right.keys()==expected.keys()
    for key,old in left.items():
        patch=right[key]
        assert old["owner"]==patch["owner"] and old["axes"]==patch["axes"]
        assert old["components"]==patch["components"]==2
        assert [(a[0],a[1]) for a in patch["axes"]]==expected[key]
        assert all(glo<=lo-1 and ghi>=hi+1 for lo,hi,glo,ghi in patch["axes"])
        a=np.asarray(old["bits"],dtype=np.uint64).view(np.float64).reshape(2,-1)
        b=np.asarray(patch["bits"],dtype=np.uint64).view(np.float64).reshape(2,-1)
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        assert np.array_equal(a[constant],np.ones(a.shape[1]))
        assert np.array_equal(b[constant],np.ones(b.shape[1]))
        evolved=1-constant
        assert np.max(np.abs(b[evolved]-a[evolved]-elapsed))<=affine_bound(np.max(np.abs(a[evolved])),elapsed,steps),key
