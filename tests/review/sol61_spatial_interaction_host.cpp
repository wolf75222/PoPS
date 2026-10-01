// Host protocol substitutes only; the consumer body is appended unchanged by pytest.
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define POPS_HD
#ifndef POPS_REAL_TYPE
#define POPS_REAL_TYPE double
#endif
using HostReal=POPS_REAL_TYPE;
namespace Kokkos {
using std::isfinite;
struct LayoutRight{}; struct HostSpace{}; struct WithoutInitializing_t{};
inline WithoutInitializing_t WithoutInitializing;
inline int view_alloc(WithoutInitializing_t,const char*) {return 0;}
template<class T,class L,class M> struct View {
  std::shared_ptr<std::vector<HostReal>> data; size_t width=0;
  View()=default; View(int,size_t n,size_t m):data(std::make_shared<std::vector<HostReal>>(n*m)),width(m){}
  HostReal& operator()(size_t i,size_t j) const {return data->at(i*width+j);}
};
template<class T> T create_mirror(const T& x) {T y=x;y.data=std::make_shared<std::vector<HostReal>>(*x.data);return y;}
template<class T> void deep_copy(T a,T b) {*a.data=*b.data;}
}
namespace pops {
using Real=HostReal;
template<int D> using Index=std::array<int,D>;
template<int D> using Extent=Index<D>;
template<int D> using RealVector=std::array<Real,D>;
template<int D> struct Box {
  Index<D> lo{},hi{}; Box()=default; Box(Index<D> a,Index<D> b):lo(a),hi(b){}
  int length(int a) const{return hi[a]-lo[a]+1;}
  size_t numPts()const{size_t n=1;for(int a=0;a<D;++a)n*=std::max(0,length(a));return n;}
  bool empty()const{return numPts()==0;}
  Box intersect(const Box& b)const{Box v=*this;for(int a=0;a<D;++a){v.lo[a]=std::max(lo[a],b.lo[a]);v.hi[a]=std::min(hi[a],b.hi[a]);}return v;}
  bool operator==(const Box&) const=default;
};
template<int D> struct Geometry {
  Box<D> box; RealVector<D> low{},up{};
  auto domain()const{return box;} auto lower()const{return low;} auto upper()const{return up;}
  double spacing(int a)const{return (up[a]-low[a])/box.length(a);}
  RealVector<D> cell_center(Index<D> i)const{RealVector<D> r{};for(int a=0;a<D;++a)r[a]=low[a]+(i[a]-box.lo[a]+.5)*spacing(a);return r;}
};
template<class T,int D> struct FieldView {
  std::shared_ptr<std::vector<HostReal>> data; Box<D> box; int width=0;
  HostReal& operator()(Index<D> i,int c)const{size_t j=0,stride=1;for(int a=0;a<D;++a){j+=(i[a]-box.lo[a])*stride;stride*=box.length(a);}return data->at(j*width+c);}
};
template<int D> struct RankSpace {size_t size()const{return 1;} size_t linear_rank(Index<D>)const{return 0;} Index<D> origin()const{return {};} Index<D> extent()const{Index<D> v;v.fill(1);return v;}};
template<int D> struct Distribution {bool replica=false; bool replicated()const{return replica;} RankSpace<D> rank_space()const{return {};} Index<D> owner(size_t)const{return {};} bool operator==(const Distribution&)const=default;};
template<int D,class M=Kokkos::HostSpace> struct MultiFab {
  struct Fab {FieldView<const Real,D> values; auto view()const{return values;}};
  using fab_type=Fab;
  std::vector<Box<D>> boxes; Distribution<D> dist; Index<D> rank{}; int width=0;std::vector<Fab> fabs;
  MultiFab()=default;
  MultiFab(std::vector<Box<D>> b,Distribution<D> d,Index<D> r,int w,Extent<D>):boxes(b),dist(d),rank(r),width(w){for(auto box:b)fabs.push_back({{std::make_shared<std::vector<HostReal>>(box.numPts()*w),box,w}});}
  const auto& layout()const{return boxes;}const auto& distribution()const{return dist;}auto local_rank()const{return rank;}
  int ncomp()const{return width;}size_t local_size()const{return fabs.size();}bool contains_local(size_t i)const{return i<fabs.size();}
  const auto& fab_global(size_t i)const{return fabs.at(i);}const auto& fab(size_t i)const{return fabs.at(i);}auto box(size_t i)const{return boxes.at(i);}
};
template<int D,class F> Real for_each_cell_reduce_sum(Box<D> b,F f){Real v=0;for(size_t n=0;n<b.numPts();++n){auto i=b.lo;size_t k=n;for(int a=0;a<D;++a){i[a]+=k%b.length(a);k/=b.length(a);}v+=f(i);}return v;}
template<int D,class F> Real for_each_cell_reduce_max(Box<D> b,F f){Real v=-INFINITY;for(size_t n=0;n<b.numPts();++n){auto i=b.lo;size_t k=n;for(int a=0;a<D;++a){i[a]+=k%b.length(a);k/=b.length(a);}v=std::max(v,f(i));}return v;}
namespace runtime::multiblock {struct Fraction {long numerator=0,denominator=1;};struct BoundaryEvaluationPoint {std::string clock="macro",graph_identity,rate_identity,application_identity;int tick=0,level=0,substep=0,stage=0;Fraction stage_fraction;double dt=1,physical_time=0;};}
namespace runtime::program {
struct ExecutionLane {int size()const{return 1;}int rank()const{return 0;}};
inline long all_reduce_max(long v,const ExecutionLane&){return v;}
inline long all_reduce_sum(long v,const ExecutionLane&){return v;}
struct ExactContractBuilder {template<class T> auto& scalar(T){return *this;}auto& text(std::string_view){return *this;}auto& presence(bool){return *this;}std::string release(){return "host-authority";}};
inline bool all_ranks_agree_exact_ordered_byte_pairs(std::initializer_list<std::pair<std::string,std::string>>,const ExecutionLane&){return true;}
// Actual FiniteCompensatedSum and full spatial_direct_interaction.hpp body follow.
}}
