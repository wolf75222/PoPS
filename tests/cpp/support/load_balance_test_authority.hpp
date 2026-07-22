#pragma once

#include <pops/parallel/prepared_load_balance.hpp>

#include <memory>

namespace pops::test {

inline std::shared_ptr<const PreparedLoadBalanceAuthority>
prepare_test_space_filling_curve_load_balance() {
  return std::make_shared<const PreparedLoadBalanceAuthority>(prepare_load_balance_authority(
      "space_filling_curve", "pops.test.amr.space-filling-curve@1",
      PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

}  // namespace pops::test
