// Standalone host receipt of the production dense primitive; no AMR/MPI runtime.
#include <pops/numerics/elliptic/interface/full_residual_basis_lu.hpp>
#include <array>
#include <iostream>
#include <numeric>

using pops::Real;
using pops::FullResidualBasisLU;
int checks = 0;
void require(bool value) {
  ++checks;
  if (!value) throw std::runtime_error("independent dense LU assertion failed");
}
template<class Exception, class Operation> void refuses(Operation operation) {
  bool caught = false;
  try { operation(); } catch (const Exception&) { caught = true; }
  require(caught);
}
void solve_permuted(const std::vector<Real>& original, const std::vector<Real>& exact,
                    const std::vector<std::size_t>& rows,
                    const std::vector<std::size_t>& columns) {
  const auto n = exact.size();
  FullResidualBasisLU lu;
  lu.allocate(n);
  std::vector<Real> rhs(n), result(n);
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      const Real a = original[rows[row]*n + columns[column]];
      lu.entry(row,column) = a;
      rhs[row] += a*exact[columns[column]];
    }
  }
  lu.factor();
  lu.apply(rhs,result);
  for (std::size_t row = 0; row < n; ++row)
    require(std::abs(result[row]-exact[columns[row]]) < Real(2e-12));
  for (std::size_t row = 0; row < n; ++row) {
    Real residual = -rhs[row];
    for (std::size_t column = 0; column < n; ++column)
      residual += original[rows[row]*n+columns[column]]*result[column];
    require(std::abs(residual) < Real(1e-10));
  }
  // A retained factor must remain stationary and allow an aliased RHS/output.
  auto aliased = rhs;
  lu.apply(aliased,aliased);
  for (std::size_t row = 0; row < n; ++row)
    require(aliased[row] == result[row]);
  lu.begin_assembly();
  refuses<std::logic_error>([&] { lu.apply(rhs,result); });
}
int main() {
  const std::vector<Real> a{0,2,1, 1,0,3, 4,5,6}; // nonsymmetric, indefinite, zero first pivot
  const std::vector<Real> exact{1,-2,3};
  std::vector<std::size_t> rows{0,1,2}, columns{0,1,2};
  do {
    columns = {0,1,2};
    do { solve_permuted(a,exact,rows,columns); }
    while (std::next_permutation(columns.begin(),columns.end()));
  } while (std::next_permutation(rows.begin(),rows.end()));
  solve_permuted({1,1000,0,0, 0,2,1000,0, 0,0,3,1000, 0,0,0,-4},
                 {1,-2,3,-4}, {0,1,2,3}, {0,1,2,3}); // strongly nonnormal, non-SPD
  FullResidualBasisLU lu;
  lu.allocate(2);
  std::array<Real,2> rhs{1,2}, output{};
  refuses<std::logic_error>([&] { lu.apply(rhs,output); });
  lu.entry(0,0)=1; lu.entry(0,1)=2; lu.entry(1,0)=2; lu.entry(1,1)=4;
  refuses<std::invalid_argument>([&] { lu.factor(); });
  refuses<std::logic_error>([&] { lu.apply(rhs,output); });
  lu.allocate(2); lu.entry(0,0)=std::numeric_limits<Real>::quiet_NaN();
  refuses<std::invalid_argument>([&] { lu.factor(); });
  lu.allocate(2); lu.entry(0,0)=std::numeric_limits<Real>::infinity();
  refuses<std::invalid_argument>([&] { lu.factor(); });
  lu.allocate(2);
  const Real large = std::numeric_limits<Real>::max()/Real(1.5);
  lu.entry(0,0)=large; lu.entry(0,1)=large;
  lu.entry(1,0)=-large; lu.entry(1,1)=large;
  refuses<std::invalid_argument>([&] { lu.factor(); });
  lu.allocate(2); lu.entry(0,0)=1; lu.entry(1,1)=1; lu.factor();
  rhs[0]=std::numeric_limits<Real>::infinity();
  refuses<std::invalid_argument>([&] { lu.apply(rhs,output); });
  refuses<std::logic_error>([&] { lu.apply(std::span<const Real>(rhs).first(1),output); });
  const auto maximum = std::numeric_limits<std::size_t>::max();
  refuses<std::length_error>([&] { (void)FullResidualBasisLU::checked_add(maximum,1); });
  refuses<std::length_error>([&] { (void)FullResidualBasisLU::checked_product(maximum,2); });
  refuses<std::length_error>([&] { (void)FullResidualBasisLU::required_bytes(maximum); });
  require(FullResidualBasisLU::required_bytes(3,37) ==
          37+9*sizeof(Real)+3*(2*sizeof(Real)+sizeof(std::size_t)));
  lu.allocate(0); lu.factor(); lu.apply({},{});
  std::cout << "independent actual-header host checks=" << checks << '\n';
}
