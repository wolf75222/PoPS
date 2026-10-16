// A target layout is an explicit capability; this allocates representation
// only.
public:
std::shared_ptr<field_type> allocate_field_storage(const field_type& prototype,
                                                   const FieldTopologyView<Dim>& topology,
                                                   double initial) const {
  return continuation_local_phase(
      [&] {
        require_field_topology(topology);
        if (topology.side() != FieldTopology::Side::Target)
          throw std::invalid_argument("Field allocation requires an exact target topology");
        const auto authority = regrid_field_qualification_(prototype);
        if (!authority || authority->block != topology.program_block() ||
            authority->provider != topology.field_provider() ||
            prototype.ncomp() != topology.components() || prototype.ghosts() != topology.ghosts())
          throw std::invalid_argument("Field allocation changed its qualified scientific template");
        if (authority->side == FieldTopology::Side::Target && authority->level != topology.level())
          throw std::invalid_argument("Field allocation template has a different target level");
        auto field = std::make_shared<field_type>(topology.layout(), topology.distribution(),
                                                  topology.local_rank(), topology.components(),
                                                  topology.ghosts());
        field->set_val(static_cast<Real>(initial));
        topology.require_field(*field);
        // Retain the exact object until this topology frame is revoked, so its
        // registered pointer cannot be recycled into another storage authority.
        require_continuation_regrid_().allocated_fields.push_back(field);
        continuation_bind_regrid_import_(*field, topology, prototype);
        return field;
      },
      "explicit target Field storage allocation");
}
