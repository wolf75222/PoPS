"""Bounded POPSAND9 wire walker, not a semantic/native authority validator.
Source: amr_program_checkpoint.hpp write_state/read_state. Unsupported pending
remaps/interface fragments fail closed; no offsets inferred from observations.
"""
import struct
class Reader:
    def __init__(self,b):self.b=b;self.i=0;self.spans=[]
    def take(self,n):
        if n<0 or self.i+n>len(self.b):raise ValueError("truncated wire")
        x=self.b[self.i:self.i+n];self.i+=n;return x
    def u(self):return struct.unpack("<Q",self.take(8))[0]
    def text(self):return self.take(self.u())
    def count(self):
        n=self.u()
        if n>len(self.b)-self.i:raise ValueError("count exceeds wire")
        return n
    def clock(self):self.take(40)
    def attempt(self,name):
        start=self.i;value=self.u();self.spans.append((name,start,self.i,value))
def spans(b):
    r=Reader(b)
    if r.take(8)!=b"POPSAND9" or struct.unpack("<q",r.take(8))[0]!=2:raise ValueError("wire/dimension")
    r.text();r.take(16);r.attempt("accepted_attempt")
    for _ in range(r.count()):r.clock()
    for _ in range(r.count()):r.text();r.take(8)
    for _ in range(r.count()):
        r.text();r.take(8)
        for _ in range(4):r.text()
        r.take(16)
    for _ in range(r.count()):r.text();r.take(16+8+8+8+32)
    if r.count()!=0:raise ValueError("pending remap unsupported")
    r.text();r.take(8);r.text();r.take(24)
    for _ in range(r.count()):r.take(32)
    r.text();r.text();r.text()
    for axis in range(2):
        for j in range(r.count()):
            r.text();r.text();r.take(16+8+8+32);r.clock();r.text();r.text();r.attempt(f"face[{axis}][{j}].key.attempt");r.take(16+48+16)
            for _ in range(r.count()):r.take(8)
    if r.count()!=0:raise ValueError("interface fragments unsupported")
    for _ in range(r.count()):r.take(24);r.text();r.clock()
    present=r.u()
    if present not in (0,1):raise ValueError("origin tag")
    if present:r.text();r.take(24)
    if r.i!=len(b):raise ValueError("trailing wire")
    return r.spans
