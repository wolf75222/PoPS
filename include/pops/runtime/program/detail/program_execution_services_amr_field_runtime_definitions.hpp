
struct GeneratedFieldRoute {
  bool prepared = false;
  std::string field;
  std::vector<int> program_blocks;
  std::vector<int> runtime_blocks;
  std::vector<const field_type*> runtime_stages;
  std::vector<const field_type*> unique_stages;
};

static facade_type* require_facade_(facade_type* facade) {
  if (facade == nullptr)
    throw std::invalid_argument("AmrStorageTopologyAdapter requires a non-null exact-ranked facade");
  return facade;
}
static runtime_type* require_runtime_(facade_type& facade) {
  return require_runtime_(facade.program_engine_());
}
static runtime_type* require_runtime_(runtime_type* runtime) {
  if (runtime == nullptr)
    throw std::invalid_argument("AmrStorageTopologyAdapter requires a materialized exact-ranked runtime");
  return runtime;
}
