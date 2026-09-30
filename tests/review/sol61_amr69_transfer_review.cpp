// Independent host reception of exact production FieldView and transfer provider.
#include <pops/amr/transfer/transfer_provider.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace pops;
using namespace pops::amr::transfer;
using pops::amr::RefinementRatio;
int checks=0;
void check(bool b){if(!b)throw std::runtime_error("independent assertion failed");++checks;}
void close(double a,double b){check(std::abs(a-b)<3e-13);}
template<class Fn>void refuses(Fn f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}check(caught);}
template<int D,class F>void cells(const Box<D>&b,F f){Index<D>i=b.lo;while(true){f(i);int a=0;for(;a<D;++a){if(i[a]<b.hi[a]){++i[a];break;}i[a]=b.lo[a];}if(a==D)break;}}
template<int D>struct Host {
 Box<D> box;int width;std::vector<double> data;
 Host(Box<D>b,int n):box(b),width(n),data(b.numPts()*n,-12345.){}
 template<class T>FieldView<T,D> make()const{
   FieldView<T,D>v{};v.data=const_cast<double*>(data.data());v.origin=box.lo;v.extents=box.extent();
   v.strides[0]=1;for(int a=1;a<D;++a)v.strides[a]=v.strides[a-1]*v.extents[a-1];
   v.ncomp=width;v.component_stride=box.numPts();return v;
 }
 auto view(){return make<double>();}auto source()const{return make<const double>();}
 double&at(Index<D>i,int c){return view()(i,c);}double at(Index<D>i,int c)const{return source()(i,c);}
};
template<int D>Box<D> refine(Box<D>b,RefinementRatio<D>r,IndexMapping<D>m){for(int a=0;a<D;++a){b.lo[a]=m.fine_origin[a]+(b.lo[a]-m.coarse_origin[a])*r[a];b.hi[a]=m.fine_origin[a]+(b.hi[a]-m.coarse_origin[a]+1)*r[a]-1;}return b;}
template<int D>double affine(Index<D>i,int c,const IndexMapping<D>&m,const RefinementRatio<D>&r,bool fine){double v=11+3*c;for(int a=0;a<D;++a){double x=fine?(i[a]-m.fine_origin[a]+.5)/r[a]:i[a]-m.coarse_origin[a]+.5;v+=(c+1)*(a+1)*.125*x;}return v;}
template<int D>void affine_case(std::array<int,D>rates){
 RefinementRatio<D>r(rates);IndexMapping<D>m;Box<D>domain;
 for(int a=0;a<D;++a){m.coarse_origin[a]=-5+7*a;m.fine_origin[a]=-17+11*a;domain.lo[a]=m.coarse_origin[a]-2;domain.hi[a]=m.coarse_origin[a]+1;}
 auto fine_box=refine(domain,r,m);Host<D>src(domain,4),dst(fine_box,5),back(domain,4);
 cells(domain,[&](auto i){for(int c=0;c<4;++c)src.at(i,c)=affine(i,c,m,r,false);});
 PhysicalParentBoundary<D>physical;physical.domain=domain;physical.lower.fill(true);physical.upper.fill(true);
 auto provider=TransferProvider<D,Centering::Cell>::linear_prolongation();
 auto plain_before=dst.data;
 refuses([&]{provider.prepare(src.source(),dst.view(),fine_box,r,m,{1,2,2});});check(dst.data==plain_before);
 auto op=provider.prepare_physical_boundary_prolongation(src.source(),dst.view(),fine_box,r,m,{1,2,2},physical);
 cells(fine_box,[&](auto i){op(i);for(int c=0;c<2;++c)close(dst.at(i,c+2),affine(i,c+1,m,r,true));close(dst.at(i,0),-12345);close(dst.at(i,1),-12345);close(dst.at(i,4),-12345);});
 auto restriction=TransferProvider<D,Centering::Cell>::conservative_restriction().prepare(dst.source(),back.view(),domain,r,m,{2,1,2});
 cells(domain,[&](auto i){restriction(i);close(back.at(i,1),src.at(i,1));close(back.at(i,2),src.at(i,2));});
 // Discrete fine gradients at both physical parent faces reproduce every slope.
 cells(fine_box,[&](auto i){for(int a=0;a<D;++a){if(i[a]==fine_box.hi[a])continue;auto j=i;++j[a];for(int c=1;c<3;++c)close((dst.at(j,c+1)-dst.at(i,c+1))*r[a],(c+1)*(a+1)*.125);}});
 // Parent-average conservation also holds for nonaffine MC-limited data.
 cells(domain,[&](auto i){for(int c=0;c<4;++c){double v=(c+1)*.3;for(int a=0;a<D;++a)v+=(c+1)*std::sin(.9*i[a]);src.at(i,c)=v;}});
 op=provider.prepare_physical_boundary_prolongation(src.source(),dst.view(),fine_box,r,m,{1,2,2},physical);
 cells(fine_box,[&](auto i){op(i);});
 cells(domain,[&](auto i){restriction(i);close(back.at(i,1),src.at(i,1));close(back.at(i,2),src.at(i,2));});
 // A periodic refined axis cannot be clipped using the other physical flags.
 for(int a=0;a<D;++a)if(r[a]>1){auto periodic=physical;periodic.lower[a]=periodic.upper[a]=false;auto before=dst.data;
 refuses([&]{provider.prepare_physical_boundary_prolongation(src.source(),dst.view(),fine_box,r,m,{1,2,2},periodic);});check(dst.data==before);}
}
double independent_mc(double lower,double center,double upper){double l=center-lower,u=upper-center;if(l*u<=0)return 0;double sign=l>0?1:-1;return sign*std::min(std::abs((l+u)/2),std::min(2*std::abs(l),2*std::abs(u)));}
void periodic_and_interpatch(){
 Box<1>domain{Index<1>{-3},Index<1>{2}},halo{Index<1>{-4},Index<1>{3}};IndexMapping<1>m{Index<1>{-3},Index<1>{-11}};RefinementRatio<1>r{3};auto fb=refine(domain,r,m);
 Host<1>s(halo,3),d(fb,3);double pi=std::acos(-1.);
 auto sample=[&](int i,int c){int wrapped=((i+3)%6+6)%6-3;return(c+1)*std::sin(2*pi*(wrapped+3+.5)/6);};
 cells(halo,[&](auto i){for(int c=0;c<3;++c)s.at(i,c)=sample(i[0],c);});
 auto p=TransferProvider<1,Centering::Cell>::linear_prolongation();PhysicalParentBoundary<1>b{domain,{false},{false}};
 auto op=p.prepare_physical_boundary_prolongation(s.source(),d.view(),fb,r,m,{0,0,3},b);
 cells(fb,[&](auto i){op(i);int offset=i[0]-m.fine_origin[0];int parent=m.coarse_origin[0]+offset/3;int child=offset%3;
   for(int c=0;c<3;++c)close(d.at(i,c),sample(parent,c)+independent_mc(sample(parent-1,c),sample(parent,c),sample(parent+1,c))*(2*child+1-3)/6.);});
 // Two neighboring destination patches receive precisely the single-patch values.
 Host<1>split(fb,3);Box<1>left{fb.lo,Index<1>{fb.lo[0]+8}},right{Index<1>{fb.lo[0]+9},fb.hi};
 auto a=p.prepare(s.source(),split.view(),left,r,m,{0,0,3});auto z=p.prepare(s.source(),split.view(),right,r,m,{0,0,3});
 cells(left,[&](auto i){a(i);});cells(right,[&](auto i){z(i);});check(split.data==d.data);
 // Interior storage edges never authorize clipping against the global physical domain.
 Host<1>partial(Box<1>{Index<1>{-1},Index<1>{0}},3),interior(Box<1>{Index<1>{-5},Index<1>{0}},3);
 PhysicalParentBoundary<1>physical{domain,{true},{true}};
 auto before=interior.data;refuses([&]{p.prepare_physical_boundary_prolongation(partial.source(),interior.view(),interior.box,r,m,{0,0,3},physical);});check(interior.data==before);
}
void singleton_and_crossing(){
 Box<1>single{Index<1>{7},Index<1>{7}},fine{Index<1>{-4},Index<1>{-1}};IndexMapping<1>m{Index<1>{7},Index<1>{-4}};RefinementRatio<1>r{4};Host<1>s(single,2),d(fine,2);
 auto p=TransferProvider<1,Centering::Cell>::linear_prolongation();PhysicalParentBoundary<1>b{single,{true},{true}};
 refuses([&]{p.prepare_physical_boundary_prolongation(s.source(),d.view(),fine,r,m,{0,0,2},b);});
 auto inject=TransferProvider<1,Centering::Cell>::constant_injection();s.at(Index<1>{7},0)=3;s.at(Index<1>{7},1)=5;
 auto op=inject.prepare(s.source(),d.view(),fine,r,m,{0,0,2});cells(fine,[&](auto i){op(i);close(d.at(i,0),3);close(d.at(i,1),5);});
 refuses([&]{inject.prepare_physical_boundary_prolongation(s.source(),d.view(),fine,r,m,{0,0,2},b);});
 Box<1>domain{Index<1>{0},Index<1>{2}},outside{Index<1>{-2},Index<1>{5}};Host<1>coarse(domain,1),dest(outside,1);PhysicalParentBoundary<1>auth{domain,{true},{true}};
 refuses([&]{p.prepare_physical_boundary_prolongation(coarse.source(),dest.view(),outside,RefinementRatio<1>{2},{},{0,0,1},auth);});
 // An unrefined singleton axis needs no one-sided stencil.
 Box<2>flat{Index<2>{4,-2},Index<2>{4,1}};IndexMapping<2>map{Index<2>{4,-2},Index<2>{13,-7}};RefinementRatio<2>ratio{1,3};Host<2>u(flat,1),v(refine(flat,ratio,map),1);
 PhysicalParentBoundary<2>phys{flat,{true,true},{true,true}};cells(flat,[&](auto i){u.at(i,0)=i[1]+.5;});
 auto kernel=TransferProvider<2,Centering::Cell>::linear_prolongation().prepare_physical_boundary_prolongation(u.source(),v.view(),v.box,ratio,map,{0,0,1},phys);
 cells(v.box,[&](auto i){kernel(i);close(v.at(i,0),-2+(i[1]+7+.5)/3);});
}
int main(){affine_case<1>({3});affine_case<2>({2,5});affine_case<3>({1,3,2});periodic_and_interpatch();singleton_and_crossing();std::cout<<checks<<" independent actual-header host checks PASS\n";}
