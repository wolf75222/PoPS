// Detached AMR adapter carriers and accepted-state staging.

template <int Dim>
struct ProgramSpatialSnapshot {
  std::string spatial_contract;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;

  bool operator==(const ProgramSpatialSnapshot&) const = default;
};

/// Bind-owned, reusable POPSAND5 assembly image.  It is deliberately distinct from the facade's
/// accepted bytes: hot refresh fills this candidate, serializes and collectively agrees it, then
/// the facade alone publishes the immutable byte image.  `valid` is reset before every fill so a
/// rejected candidate can never be reused by a later publication.
template <int Dim>
struct AcceptedStateStaging {
  using face_fragment_type = ::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>;
  using interface_fragment_type = ::pops::amr::InterfaceFluxFragment<AmrProgramFacePayload>;
  using interface_ledger_type =
      ::pops::amr::TransactionalInterfaceFluxLedger<AmrProgramFacePayload>;
  /// Sortable non-owning carrier for one dense ledger fragment.  The ledger's FragmentView holds
  /// its measure by reference, which intentionally prevents assigning it into a vector; this
  /// staging view owns only that trivially copyable measure and retains all text/payload as
  /// stable views into the immutable published dense image.
  struct InterfaceFluxSerializationView {
    typename interface_ledger_type::FragmentKeyView key;
    ::pops::amr::InterfaceFluxFragmentMeasure measure;
    std::span<const Real> payload;
    /// The ledger-owned dense source slot and checkpoint wire position are identical only after
    /// the canonical-order check in the adapter.  Keep the value explicit so a copied staging
    /// image can never infer an ordering from text keys during Candidate.
    std::size_t wire_ordinal = 0;
  };
  struct HistorySlotBinding {
    std::string key;
    std::size_t state_slot = 0;
    std::size_t source_slot = 0;
  };

  AmrProgramAcceptedState<Dim> state;
  /// Preindexed at bind.  Refresh follows these resident map keys without reparsing history
  /// strings or rebuilding descriptors/sets in the accepted-step path.
  std::vector<HistorySlotBinding> history_slot_bindings;
  /// Configured-depth history envelope.  The logical checkpoint vector exchanges its nested
  /// names with this pool as active hierarchy levels grow or shrink.
  std::vector<AmrProgramHistorySlotProvenance> history_slot_pool;
  std::vector<std::size_t> history_slot_active_indices;
  std::vector<std::string> pending_history_keys;
  std::vector<AmrProgramPendingHistoryRemap> pending_history_remap_slots;
  std::vector<std::size_t> pending_history_active_slots;
  std::array<std::vector<face_fragment_type>, Dim> accepted_face_flux_slots;
  std::array<std::vector<const face_fragment_type*>, Dim> accepted_face_flux_sources;
  std::array<std::vector<std::size_t>, Dim> accepted_face_flux_active_slots;
  std::vector<AmrProgramSynchronizationEvent> synchronization_event_slots;
  std::vector<std::size_t> synchronization_event_active_indices;
  std::vector<interface_fragment_type> accepted_interface_flux_slots;
  std::vector<std::size_t> accepted_interface_flux_active_slots;
  bool prepared_envelope = false;
  bool primed = false;
  bool valid = false;
  std::size_t configured_level_count = 0;
  std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();
};

/// One Program specialization over one immutable native rank.
///
/// The context never decodes a dimension tag and never pads an absent axis.  Its active level is a
/// compile-time-ranked `MultiFab<Dim>` selected from the exact `AmrRuntime<Dim>` hierarchy.  The
/// retained generated block owns geometry, physical boundaries, same-level/coarse-fine ghost fill,
/// residual assembly and integrated face fluxes.  Unsupported provider families fail before a valid
/// cell is changed.
