// Native representation adapters. The generated equations own all transfer
// laws.
public:
ContinuationValueRef<Dim> export_field_bundle(const field_type& field,
                                              const FieldTopologyView<Dim>& topology,
                                              const ContinuationType& schema,
                                              std::uint64_t byte_capacity) const {
  return field_bundle::export_field<Dim>(*this, field, topology, schema, byte_capacity);
}
std::shared_ptr<field_type> import_field_bundle(const ContinuationValueRef<Dim>& bundle,
                                                const field_type& prototype,
                                                const FieldTopologyView<Dim>& topology,
                                                const ContinuationType& schema,
                                                std::uint64_t byte_capacity) const {
  auto field =
      field_bundle::import_field<Dim>(*this, bundle, prototype, topology, schema, byte_capacity);
  continuation_local_phase([&] { continuation_bind_regrid_import_(*field, topology); },
                           "imported Field topology authority");
  return field;
}
