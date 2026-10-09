using program_value_slot_type = typename ProgramValueAuthority<Dim>::Slot;

int program_value_level_() const noexcept { return 0; }
std::string program_value_identity_() const { return system_->installed_program_hash(); }
void revoke_program_history_values_() const noexcept {
  for (const auto& [name,ring] : runtime_state().hist_.histories) {
    (void)name;
    for (const auto& field : ring) program_values_.revoke_buffer_(field);
  }
}

program_value_slot_type program_value_slot_(int program_block, const field_type& field) const {
  const int owner = sys_block(program_block);
  if (&field == &system_->block_state(owner))
    return {ProgramValueStorage::State,0,owner,-1,0,{},0};
  for (const auto& [key, scratch] : scratch_) {
    if (&field != &scratch) continue;
    if (std::get<1>(key) != owner)
      throw std::invalid_argument("Program SSA scratch belongs to another runtime block");
    const auto kind = std::get<0>(key) == ScratchKind::Rhs ? ProgramValueStorage::RhsScratch :
        std::get<0>(key) == ScratchKind::State ? ProgramValueStorage::StateScratch :
                                               ProgramValueStorage::ScalarScratch;
    return {kind,0,owner,std::get<2>(key),std::get<3>(key),{},0};
  }
  const auto& histories = runtime_state().hist_;
  for (const auto& [name, ring] : histories.histories)
    for (std::size_t lag = 0; lag < ring.size(); ++lag)
      if (&field == &ring[lag]) {
        if (histories.owner.at(name) != owner)
          throw std::invalid_argument("Program SSA history belongs to another runtime block");
        return {ProgramValueStorage::History,0,owner,-1,0,name,static_cast<int>(lag)};
      }
  throw std::invalid_argument("Program SSA buffer has no context-owned native storage slot");
}

int scratch_prototype_owner_(const field_type& field) const {
  for (int block = 0; block < system_->n_blocks(); ++block)
    if (&field == &system_->block_state(block)) return block;
  for (const auto& [key, scratch] : scratch_)
    if (&field == &scratch) return std::get<1>(key);
  const auto& histories = runtime_state().hist_;
  for (const auto& [name, ring] : histories.histories)
    for (const auto& slot : ring)
      if (&field == &slot) return histories.owner.at(name);
  throw std::invalid_argument("Program scratch prototype has no context-owned runtime block");
}
