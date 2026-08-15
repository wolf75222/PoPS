#pragma once

#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/core/foundation/native_dimension.hpp>

#include <memory>

namespace pops::test {

template <int Dim = kNativeDimension>
inline std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>>
prepare_test_space_filling_curve_load_balance() {
  return std::make_shared<const PreparedLoadBalanceAuthority<Dim>>(
      prepare_load_balance_authority<Dim>(
          "space_filling_curve", "pops.test.amr.space-filling-curve@1",
          PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

}  // namespace pops::test
