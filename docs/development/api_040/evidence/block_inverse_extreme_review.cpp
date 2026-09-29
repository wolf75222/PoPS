// Independent host-only adversarial receipt. No native runtime or PDE claim.
// Compile directly with -std=c++20 -O2 -Iinclude (optionally ASan/UBSan).
#include <pops/numerics/linalg/block_inverse.hpp>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

using pops::Real;
namespace {
int failures = 0, checks = 0;
void require(bool pass, const std::string& label) {
  ++checks;
  if (!pass) { ++failures; std::fprintf(stderr, "FAIL %s\n", label.c_str()); }
}
bool close(Real actual, Real expected) {
  if (!std::isfinite(actual)) return false;
  return expected == 0 ? actual == 0 : std::abs(actual / expected - 1) < 4.e-13;
}
template<int N> void scaled_dense(const Real (&scale)[N], bool permuted) {
  // Independent analytic oracle: A = D (I + alpha 11^T) P.
  // A^-1 = P^T (I - alpha/(1+N alpha) 11^T) D^-1.
  constexpr Real alpha = .125;
  const Real beta = alpha / (1 + N * alpha);
  Real matrix[N][N], inverse[N][N], rhs[N], solution[N];
  for (int r=0;r<N;++r) {
    rhs[r] = scale[r] * (1 + N*alpha) * .125;
    solution[r] = -777;
    for (int c=0;c<N;++c) {
      const int unpermuted = permuted ? (c+1)%N : c;
      matrix[r][c] = scale[r] * ((r==unpermuted ? 1. : 0.) + alpha);
      inverse[r][c] = -777;
    }
  }
  const std::string name = "dense N="+std::to_string(N)+(permuted?" permuted":"");
  bool ok=pops::detail::block_inverse<N>(matrix,inverse);
  require(ok,name+" inverse accepted");
  if (ok) for(int r=0;r<N;++r) for(int c=0;c<N;++c) {
    const int original_row = permuted ? (r+1)%N : r;
    require(close(inverse[r][c],((original_row==c ? 1. : 0.)-beta)/scale[c]),
            name+" inverse entry "+std::to_string(r)+","+std::to_string(c));
  }
  ok=pops::detail::block_apply_inverse<N>(matrix,rhs,solution);
  require(ok,name+" solve accepted");
  if(ok) for(int i=0;i<N;++i) require(close(solution[i],.125),name+" solve component");
}
template<int N> void diagonal_extremes() {
  for(Real scale : {Real(1e308),Real(1e-308)}) {
    Real a[N][N]{},inverse[N][N]{},rhs[N],out[N];
    for(int i=0;i<N;++i) { a[i][i]=scale;rhs[i]=scale*.25;out[i]=-777; }
    const std::string name="diagonal N="+std::to_string(N)+" scale="+std::to_string(std::log10(scale));
    const bool ok=pops::detail::block_inverse<N>(a,inverse);
    require(ok,name+" accepted");
    if(ok) for(int i=0;i<N;++i) for(int j=0;j<N;++j)
      require(close(inverse[i][j],i==j?1/scale:0),name+" inverse");
    const bool solved=pops::detail::block_apply_inverse<N>(a,rhs,out);
    require(solved,name+" solve accepted");
    if(solved) for(int i=0;i<N;++i) require(close(out[i],.25),name+" solve");
  }
}
template<int N> void reject_without_publication() {
  Real matrix[N][N]{},inverse[N][N],rhs[N],out[N];
  for(int i=0;i<N;++i) {
    rhs[i]=1;out[i]=-777;
    for(int j=0;j<N;++j) inverse[i][j]=-777;
    if(i+1<N) matrix[i][i]=1;
  }
  require(!pops::detail::block_inverse<N>(matrix,inverse),"singular inverse refusal");
  for(auto& row:inverse) for(Real v:row) require(v==-777,"singular inverse sentinel");
  require(!pops::detail::block_apply_inverse<N>(matrix,rhs,out),"singular solve refusal");
  for(Real v:out) require(v==-777,"singular solve sentinel");
  matrix[N-1][N-1]=std::numeric_limits<Real>::infinity();
  require(!pops::detail::block_inverse<N>(matrix,inverse),"infinite matrix refusal");
  for(auto& row:inverse) for(Real v:row) require(v==-777,"infinite matrix sentinel");
  matrix[N-1][N-1]=1;
  rhs[N-1]=std::numeric_limits<Real>::quiet_NaN();
  require(!pops::detail::block_apply_inverse<N>(matrix,rhs,out),"NaN rhs refusal");
  for(Real v:out) require(v==-777,"NaN rhs sentinel");
}
template<int N> void cancellation() {
  Real matrix[N][N]{},rhs[N],out[N];
  for(int i=0;i<N;++i) { matrix[i][i]=1;rhs[i]=1e308;out[i]=-777; }
  // inverse row0 = [1,1,-1,0...], exact solution all 1e308.
  matrix[0][1]=-1;matrix[0][2]=1;
  const bool ok=pops::detail::block_apply_inverse<N>(matrix,rhs,out);
  require(ok,"finite cancellation accepted N="+std::to_string(N));
  if(ok) for(Real value:out) require(close(value,1e308),"finite cancellation value");
}
template<int N> void solve_without_representable_inverse() {
  Real matrix[N][N]{},inverse[N][N],rhs[N],out[N];
  for(int i=0;i<N;++i) {
    matrix[i][i]=1e-320;rhs[i]=1e-320;out[i]=-777;
    for(int j=0;j<N;++j) inverse[i][j]=-777;
  }
  require(!pops::detail::block_inverse<N>(matrix,inverse),"unrepresentable inverse refused");
  for(auto& row:inverse) for(Real v:row) require(v==-777,"unrepresentable inverse sentinel");
  const bool ok=pops::detail::block_apply_inverse<N>(matrix,rhs,out);
  require(ok,"representable solve accepted without representable inverse");
  if(ok) for(Real v:out) require(close(v,1),"subnormal matrix finite solution");
}
void unrepresentable_solution_preserves_output() {
  const Real matrix[2][2]={{.5,0},{0,1}},rhs[2]={1e308,1};
  Real out[2]={-777,-777};
  require(!pops::detail::block_apply_inverse<2>(matrix,rhs,out),"unrepresentable solution refused");
  require(out[0]==-777 && out[1]==-777,"unrepresentable solution not partly published");
}
void apply_aliases_rhs() {
  const Real matrix[2][2]={{1,-1},{1,1}};
  Real rhs[2]={3,1};
  require(pops::detail::block_apply_inverse<2>(matrix,rhs,rhs),"in-place apply accepted");
  require(rhs[0]==2 && rhs[1]==-1,"in-place apply reads the complete old rhs");
}
void tolerance_and_parity() {
  for(Real scale0 : {Real(1e-150),Real(1),Real(1e150)})
    for(Real scale1 : {Real(1e-150),Real(1),Real(1e150)}) {
      Real a[2][2]={{scale0,scale0},{scale1,scale1*(1+1e-12)}};
      Real inv[2][2]={{-777,-777},{-777,-777}};
      require(!pops::detail::block_inverse<2>(a,inv,1e-10),"relative tolerance invariant under row scaling");
      for(auto& row:inv) for(Real v:row) require(v==-777,"relative tolerance refusal sentinel");
    }
  for(Real w : {Real(0),Real(.125),Real(-.7),Real(2),Real(100)}) {
    const Real a[2][2]={{1,-w},{w,1}},rhs[2]={.375,-.625};
    Real inv[2][2],out[2];const Real det=1+w*w,reciprocal=1/det;
    require(pops::detail::block_inverse<2>(a,inv),"ordinary inverse accepted");
    require(inv[0][0]==1/det && inv[0][1]==w/det && inv[1][0]==-w/det && inv[1][1]==1/det,
            "ordinary pinned inverse parity");
    require(pops::detail::block_apply_inverse<2>(a,rhs,out),"ordinary apply accepted");
    require(out[0]==reciprocal*(rhs[0]+w*rhs[1]) && out[1]==reciprocal*(rhs[1]-w*rhs[0]),
            "ordinary pinned apply parity");
  }
}
}
int main() {
  static_assert(pops::kRealIsBinary64,"this receipt targets binary64 extremes");
  diagonal_extremes<1>();diagonal_extremes<2>();diagonal_extremes<3>();diagonal_extremes<5>();
  Real two[2]={1e308,1e-308},three[3]={1e-308,1e308,1},five[5]={1e308,1e-308,1e150,1e-150,1};
  for(bool permutation:{false,true}) {scaled_dense(two,permutation);scaled_dense(three,permutation);scaled_dense(five,permutation);}
  reject_without_publication<2>();reject_without_publication<3>();reject_without_publication<5>();
  cancellation<3>();cancellation<5>();
  solve_without_representable_inverse<2>();solve_without_representable_inverse<3>();
  solve_without_representable_inverse<5>();unrepresentable_solution_preserves_output();
  apply_aliases_rhs();tolerance_and_parity();
  std::printf("%d checks, %d failures\n",checks,failures);
  return failures ? 1 : 0;
}
