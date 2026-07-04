// ADC-514 native per-block runtime-param facade methods for the AmrSystem binding TU.
//
// Binding-private include (not a public header): amr_system.cpp textually pulls it in AFTER
// AmrSystem::Impl is complete, so the out-of-line definitions below see p_->block_params_ and
// p_->block_index_or_throw. Split out only to keep amr_system.cpp on its frozen architecture
// budget (tests/python/architecture/test_no_legacy_runtime_routes.py); logic is unchanged.
#ifndef POPS_PYTHON_BINDINGS_AMR_AMR_SYSTEM_PARAMS_HPP_
#define POPS_PYTHON_BINDINGS_AMR_AMR_SYSTEM_PARAMS_HPP_

// NO namespace wrapper: the include site in amr_system.cpp is already INSIDE `namespace pops`
// (re-opening it here would define pops::pops::AmrSystem and fail to compile).

// Register block @p name's SHARED runtime-param vector so set_block_params resolves it by name
// pre-build; the build closures capture the vector and re-inject it each macro-step (ADC-514).
void AmrSystem::register_block_params(const std::string& name,
                                      std::shared_ptr<std::vector<double>> values) {
  p_->block_params_[name] = std::move(values);
}

// Overwrite block @p name's SHARED runtime-param values in place; the change reaches the captured
// closures at the next step WITHOUT recompiling (VERBATIM mirror of System::set_block_params).
void AmrSystem::set_block_params(const std::string& name, const std::vector<double>& values) {
  (void)p_->block_index_or_throw(name);  // same "no block named" diagnostic as density/mass
  auto it = p_->block_params_.find(name);
  if (it == p_->block_params_.end())
    throw std::runtime_error(
        "AmrSystem::set_block_params : block '" + name +
        "' has no runtime parameter (declare dsl.Param(..., kind='runtime') and wire via a "
        "production block ; const params are frozen at compile time)");
  std::vector<double>& pv = *it->second;
  if (values.size() != pv.size())
    throw std::runtime_error("AmrSystem::set_block_params : block '" + name + "' expects " +
                             std::to_string(pv.size()) + " runtime parameters, received " +
                             std::to_string(values.size()));
  pv = values;  // shared with the closures (shared_ptr): effect at the next step
}

#endif  // POPS_PYTHON_BINDINGS_AMR_AMR_SYSTEM_PARAMS_HPP_
