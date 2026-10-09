"""Paired actual-header host preparation; never a PoPS backend receipt."""
import argparse
import hashlib
import json
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CPP = r'''
#include "model.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
using State=std::array<double,15>;
State gaussian(double rho,double u,double v,double t) {
  double x[5]={1,u,u*u+t,u*u*u+3*u*t,u*u*u*u+6*u*u*t+3*t*t};
  double y[5]={1,v,v*v+t,v*v*v+3*v*t,v*v*v*v+6*v*v*t+3*t*t};
  State a{}; int k=0; for(int j=0;j<=4;++j) for(int i=0;i<=4-j;++i) a[k++]=rho*x[i]*y[j];return a;
}
int main(){
  std::cout<<std::setprecision(17);
  test_fan_li15::Kernel law;
  std::array<State,32> states;
  for(int i=0;i<32;++i){double rho=i%3==0?1.0:(i%3==1?2.0:1.0+1e-12);
    auto first=gaussian(.6*rho,-.3+.02*i,.1-.005*i,.4+.01*i);
    auto second=gaussian(.4*rho,.5-.01*i,-.2+.003*i,.8-.005*i);
    for(int k=0;k<15;++k)states[i][k]=first[k]+second[k];}
  for(int i=0;i<32;++i){ auto a=states[i],b=states[(i+7)%32]; std::array<double,2> g{i%2?.7:-.7,-.2};
    std::cout<<"input "<<i;for(double v:a)std::cout<<' '<<v;for(double v:b)std::cout<<' '<<v;for(double v:g)std::cout<<' '<<v;std::cout<<'\n';
    auto f=law.path_directional_flux(a,g); auto z=law.path_integral(a,b,g); auto r=law.path_integral(b,a,g); auto q=law.path_integral(a,b,{0.,0.});
    std::cout<<"case "<<i<<' '<<int(f.status)<<' '<<int(z.status)<<' '<<z.speed_bound;
    for(int k=0;k<15;++k){if(z.integral[k]!=-r.integral[k]||q.integral[k]!=0.)return 2;std::cout<<' '<<f.flux.values[k]<<' '<<z.integral[k];}std::cout<<'\n';
  }
  State bad=states[0];bad[0]=-1;auto neg=law.path_integral(bad,states[1],{1.,0.});
  bad=states[0];bad[5]=std::numeric_limits<double>::quiet_NaN();auto nan=law.path_integral(bad,states[1],{1.,0.});
  auto dir=law.path_integral(states[0],states[1],{std::numeric_limits<double>::infinity(),0.});
  std::cout<<"refusals "<<int(neg.status)<<' '<<int(nan.status)<<' '<<int(dir.status)<<'\n';
  for(int rep=0;rep<7;++rep){double checksum=0.;auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<30000;++i){int k=(i+rep)%32;auto z=law.path_integral(states[k],states[(k+7)%32],{.7,-.2}); auto f=law.path_directional_flux(states[k],{.7,-.2});checksum+=z.integral[12]+f.flux.values[14]+z.speed_bound;}
    double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count()/30000.;std::cout<<"timing "<<rep<<' '<<ns<<' '<<checksum<<'\n';}
}
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    refs={'old':'8d438518','new':'5000328'};receipt={'scope':'HOST_ONLY_NOT_NATIVE_SCI_OR_PERFORMANCE_QUALIFICATION','variants':{}}
    for label,ref in refs.items():
        work=out/label;work.mkdir();overlay=work/'include';overlay.mkdir()
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',ref,'include/pops/numerics/moments'],cwd=ROOT,text=True).splitlines()
        pins={}
        for name in names:
            raw=subprocess.check_output(['git','show',f'{ref}:{name}'],cwd=ROOT);p=overlay/Path(name).relative_to('include');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);pins[name]=hashlib.sha256(raw).hexdigest()
        raw=subprocess.check_output(['git','show',f'{ref}:tests/cpp/support/generated_fan_li15.hpp'],cwd=ROOT);(work/'model.hpp').write_bytes(raw);pins['generated_fan_li15.hpp']=hashlib.sha256(raw).hexdigest()
        (work/'paired.cpp').write_text(CPP)
        command=[compiler,'-std=c++20','-O2','-DPOPS_NATIVE_DIM=2','-fno-fast-math','-ffp-contract=off','-I',str(overlay),'-I',str(ROOT/'include'),str(work/'paired.cpp'),'-o',str(work/'paired')]
        result=subprocess.run(command,capture_output=True,text=True);(work/'compile.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
        result=subprocess.run([str(work/'paired')],capture_output=True,text=True);(work/'raw.txt').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
        cases=[];timings=[];inputs=[]
        for line in result.stdout.splitlines():
            bits=line.split()
            if bits[0]=='case':cases.append([int(bits[1]),int(bits[2]),int(bits[3]),*map(float,bits[4:])])
            elif bits[0]=='timing':timings.append(list(map(float,bits[1:])))
            elif bits[0]=='input':inputs.append(list(map(float,bits[2:])))
            elif bits[0]=='refusals':refusals=list(map(int,bits[1:]))
        receipt['variants'][label]={'source_ref':ref,'header_pins':pins,'command':command,'cases':cases,'inputs':inputs,'timings':timings,'refusals':refusals,'raw_sha256':hashlib.sha256((work/'raw.txt').read_bytes()).hexdigest()}
    a,b=(receipt['variants'][key] for key in ('old','new'))
    assert a['refusals']==b['refusals']
    diffs=[]
    for x,y in zip(a['cases'],b['cases'],strict=True):
        assert x[:3]==y[:3]
        diffs.extend(abs(u-v)/(1+abs(u)) for u,v in zip(x[3:],y[3:],strict=True))
    spec=importlib.util.spec_from_file_location('independent_fan_li_oracle',ROOT/'tests/python/support/fan_li15_oracle.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    def gauss(n):
        points=[]
        for i in range(1,n+1):
            z=math.cos(math.pi*(i-.25)/(n+.5))
            for iteration in range(30):
                p0,p1=1.,z
                for k in range(2,n+1):p0,p1=p1,((2*k-1)*z*p1-(k-1)*p0)/k
                dp=n*(z*p1-p0)/(z*z-1);nextz=z-p1/dp
                if abs(nextz-z)<2e-16:z=nextz;break
                z=nextz
            points.append(((1+z)/2,1/((1-z*z)*dp*dp)))
        return points
    def integrate(left,right,g,n):
        delta=[y-x for x,y in zip(left,right,strict=True)];terms=[[] for _ in range(15)]
        for t,w in gauss(n):
            values=oracle.primary_product([x+t*d for x,d in zip(left,delta,strict=True)],delta,g)
            for k,value in enumerate(values):terms[k].append(w*value)
        return [math.fsum(values) for values in terms]
    independent=[]
    for row,inputs in zip(b['cases'],b['inputs'],strict=True):
        left,right,g=inputs[:15],inputs[15:30],inputs[30:]
        flux=oracle.derivative_gaussian_flux(left,g);q32=integrate(left,right,g,32);q64=integrate(left,right,g,64)
        independent.append({'case':row[0],'flux_max_scaled_error':max(abs(row[4+2*k]-flux[k])/(1+abs(flux[k])) for k in range(15)),
            'integral64_max_scaled_error':max(abs(row[5+2*k]-q64[k])/(1+abs(q64[k])) for k in range(15)),
            'quadrature32_64_max_scaled_gap':max(abs(x-y)/(1+abs(y)) for x,y in zip(q32,q64,strict=True))})
    receipt['independent_primary_formula_float_quadrature']=independent
    import statistics
    receipt['comparison']={'max_absolute_scaled_difference':max(diffs),'bit_exact_numeric_cases':all(d==0 for d in diffs),'old_median_ns_flux_plus_integral':statistics.median(t[1] for t in a['timings']),'new_median_ns_flux_plus_integral':statistics.median(t[1] for t in b['timings']),'reversal_and_zero_exact_each_variant':True,'status_refusals_equal':True,'repetitions':7,'calls_per_repetition':30000}
    receipt['comparison']['new_over_old_cost']=receipt['comparison']['new_median_ns_flux_plus_integral']/receipt['comparison']['old_median_ns_flux_plus_integral']
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt['comparison'],sort_keys=True))
if __name__=='__main__':main()
