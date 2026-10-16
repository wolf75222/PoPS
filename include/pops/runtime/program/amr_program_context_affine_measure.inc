// Immutable physical measures for one affine macro transaction. A changing
// metric needs an authored temporal transport law; it is never silently frozen.
private:
struct AffineMeasureSnapshot {
  Geometry<Dim> geometry;
  std::optional<field_type> active, fraction;
  field_type coverage;
};
using AffineMeasureTower = std::vector<std::vector<AffineMeasureSnapshot>>;

AffineMeasureTower capture_affine_measures_() const {
  const auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
  AffineMeasureTower result(hierarchy.block_count());
  for (std::size_t owner = 0; owner < hierarchy.block_count(); ++owner)
    for (std::size_t level = 0; level < hierarchy.level_count(); ++level) {
      const auto* active = facade_->prepared_amr_block_level_active_mask(owner, level);
      const auto* fraction = facade_->prepared_amr_block_level_volume_fraction(owner, level);
      result[owner].push_back({facade_->prepared_amr_level_geometry(level),
                               active ? std::optional<field_type>(*active) : std::nullopt,
                               fraction ? std::optional<field_type>(*fraction) : std::nullopt,
                               facade_->prepared_amr_block_level_coverage_mask(owner, level)});
    }
  return result;
}

void require_fixed_affine_measure_(const field_type& before, const field_type& after) const {
  require_same_field_contract_(before, after, "affine physical measure");
  if (before.ncomp() != 1)
    throw std::invalid_argument("affine physical measure requires one scalar per cell");
  for (std::size_t local = 0; local < before.local_size(); ++local)
    if (pops::for_each_cell_reduce_max(
            before.box(local), detail::FixedMeasureDifferenceKernel<Dim>{
                                   before.fab(local).view(), after.fab(local).view()}) > Real{0})
      throw std::invalid_argument(
          "changing physical measure requires an authored temporal metric law");
}

void validate_affine_measures_() const {
  const auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
  const auto& measures = affine_finalization_->measures;
  if (measures.size() != hierarchy.block_count())
    throw std::invalid_argument("affine physical measure changed its block topology");
  for (std::size_t owner = 0; owner < measures.size(); ++owner) {
    if (measures[owner].size() != hierarchy.level_count())
      throw std::invalid_argument("affine physical measure changed its level topology");
    for (std::size_t level = 0; level < measures[owner].size(); ++level) {
      const auto& old = measures[owner][level];
      if (!(old.geometry == facade_->prepared_amr_level_geometry(level)))
        throw std::invalid_argument(
            "changing Cartesian geometry requires an authored temporal metric law");
      const auto* active = facade_->prepared_amr_block_level_active_mask(owner, level);
      const auto* fraction = facade_->prepared_amr_block_level_volume_fraction(owner, level);
      if (bool(old.active) != bool(active) || bool(old.fraction) != bool(fraction))
        throw std::invalid_argument("affine physical measure changed its mask authority");
      if (active)
        require_fixed_affine_measure_(*old.active, *active);
      if (fraction)
        require_fixed_affine_measure_(*old.fraction, *fraction);
      require_fixed_affine_measure_(old.coverage,
                                    facade_->prepared_amr_block_level_coverage_mask(owner, level));
    }
  }
}
