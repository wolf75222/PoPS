// Native representation adapters. The generated equations own all transfer
// laws.
public:
ContinuationValueRef<Dim> export_field_bundle(const field_type& field,
                                              const FieldTopologyView<Dim>& topology,
                                              const ContinuationType& schema,
                                              std::uint64_t byte_capacity) const {
  continuation_local_phase(
      [&] {
        require_regrid_field_topology(field, topology);
        if (continuation_source_field_authority_(field).representation)
          throw std::invalid_argument(
              "Field Bundle export requires its explicit representation transformation");
      },
      "exported Field source authority");
  return field_bundle::export_field<Dim>(*this, field, topology, schema, byte_capacity);
}
std::shared_ptr<field_type> import_field_bundle(const ContinuationValueRef<Dim>& bundle,
                                                const field_type& prototype,
                                                const FieldTopologyView<Dim>& topology,
                                                const ContinuationType& schema,
                                                std::uint64_t byte_capacity) const {
  continuation_local_phase(
      [&] {
        require_field_topology(topology);
        const auto authority = regrid_field_qualification_(prototype);
        if (topology.side() != FieldTopology::Side::Target || !authority ||
            authority->block != topology.program_block() ||
            authority->provider != topology.field_provider() ||
            prototype.ncomp() != topology.components() || prototype.ghosts() != topology.ghosts() ||
            (authority->side == FieldTopology::Side::Target &&
             authority->level != topology.level()))
          throw std::invalid_argument("Field import changed its qualified scientific template");
      },
      "imported Field scientific authority before allocation");
  auto field =
      field_bundle::import_field<Dim>(*this, bundle, prototype, topology, schema, byte_capacity);
  continuation_local_phase(
      [&] {
        require_continuation_regrid_().allocated_fields.push_back(field);
        continuation_bind_regrid_import_(*field, topology, prototype);
      },
      "imported Field topology authority");
  return field;
}
