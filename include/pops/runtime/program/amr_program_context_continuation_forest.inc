// Included inside AmrProgramContext. The parent and every level image form one
// accepted transaction. All scientific transformations are authored Program code.
public:
using continuation_authority_type = ContinuationAuthority<Dim>;
using continuation_ref = ContinuationValueRef<Dim>;

bool continuation_parent_phase() const noexcept {
  return continuation_parent_phase_;
}

void install_continuation_forest(ContinuationConfiguration root,
                                 ContinuationConfiguration child) const {
  continuation_local_phase(
      [&] {
        root.validate();
        child.validate();
        if (!root.fields.empty() || child.fields.size() != static_cast<std::size_t>(n_blocks()))
          throw std::invalid_argument(
              "AMR forest must declare unique globals and every local State");
        if (continuation_regrid_) {
          auto& frame = require_continuation_regrid_();
          frame.targets.clear();
          for (int level = 0; level < nlev(); ++level) {
            const auto old = frame.sources.find(level);
            auto target = old == frame.sources.end() ? ContinuationStore<Dim>{} : old->second;
            target.configure(child);
            frame.targets.emplace(level, std::move(target));
          }
          return;
        }
        auto prepared = amr_continuation_accepted_;
        prepared[-1].configure(root);
        for (int level = 0; level < nlev(); ++level)
          prepared[level].configure(child);
        amr_continuation_accepted_.swap(prepared);
      },
      "AMR complete forest schema installation");
}

void begin_continuation_forest(double duration, std::initializer_list<int> blocks) const {
  continuation_local_phase(
      [&] {
        continuation_parent_phase_ = false;
        continuation_parent_authority_.revoke();
        const auto root = amr_continuation_accepted_.find(-1);
        continuation_parent_working_ =
            root == amr_continuation_accepted_.end() ? ContinuationStore<Dim>{} : root->second;
        continuation_children_.clear();
      },
      "AMR forest immutable parent base");
  begin_level_continuation_attempt(facade_->time(), facade_->time() + duration);
  begin_affine_finalization(duration, blocks);
}

void begin_continuation_parent() const {
  if (!continuation_regrid_phase())
    prepare_affine_finalization();
  continuation_local_phase(
      [&] {
        const int levels = nlev();
        if (level_continuation_pending_.size() != static_cast<std::size_t>(levels))
          throw std::logic_error("AMR parent requires every synchronized child exactly once");
        std::vector<typename continuation_authority_type::Accepted> children;
        children.reserve(static_cast<std::size_t>(levels));
        for (int level = 0; level < levels; ++level) {
          const auto& child = level_continuation_pending_.at(level);
          // Authentication occurs before any parent observation or scientific vote.
          (void)child.accepted.value();
          children.push_back(child.accepted);
          if (!continuation_regrid_phase()) {
            const auto& store = level_continuation_macro_bases_.at(level);
            for (const auto& binding : store.configuration().fields)
              affine_finalization_->provisional.at(sys_block(binding.program_block)).at(level) =
                  continuation_at<Dim>(child.image.value, binding.path)->field();
          }
        }
        continuation_children_.swap(children);
        continuation_parent_authority_.begin(level_continuation_macro_attempt());
        continuation_parent_phase_ = true;
      },
      "AMR complete child forest binding");
}

continuation_authority_type& continuation_authority() const {
  return continuation_parent_phase_ ? continuation_parent_authority_
                                    : level_continuation_authority();
}
std::span<const typename continuation_authority_type::Accepted> continuation_child_dependencies()
    const {
  if (!continuation_parent_phase_)
    return {};
  return continuation_children_;
}
void continuation_control_consensus() const {
  for (const auto word : continuation_authority().logical_state())
    continuation_word_agrees_(word, "AMR continuation control differs across ranks");
}
std::uint64_t continuation_byte_capacity() const {
  if (continuation_regrid_phase() && !continuation_parent_phase_)
    return require_continuation_regrid_().targets.at(active_level_).configuration().byte_capacity;
  return continuation_parent_phase_ ? continuation_parent_working_.configuration().byte_capacity
                                    : level_continuation_byte_capacity();
}
void continuation_configure(ContinuationConfiguration configuration) const {
  if (!continuation_parent_phase_) {
    level_continuation_configure(std::move(configuration));
    return;
  }
  continuation_local_phase(
      [&] {
        configuration.validate();
        if (!configuration.fields.empty())
          throw std::invalid_argument("AMR parent globals cannot duplicate child State ownership");
        continuation_parent_working_.configure(std::move(configuration));
      },
      "AMR global parent configuration");
}
bool continuation_initialized() const {
  if (!continuation_parent_phase_)
    return level_continuation_initialized();
  const bool initialized = continuation_parent_working_.initialized();
  continuation_word_agrees_(initialized ? 1 : 0, "AMR parent initialization differs across ranks");
  return initialized;
}
void continuation_initialize(continuation_ref initial) const {
  if (!continuation_parent_phase_) {
    level_continuation_initialize(std::move(initial));
    return;
  }
  auto& store = continuation_parent_working_;
  auto prepared = level_continuation_prepare_(store, std::move(initial), true);
  auto keep = continuation_local_phase(
      [&] {
        level_continuation_require_frontier_(store, prepared.value, level_continuation_macro_start_,
                                             static_cast<std::int64_t>(facade_->macro_step()));
        return level_continuation_collect_retention_(store, &prepared);
      },
      "AMR global parent initial frontier");
  level_continuation_image_consensus_(store, prepared);
  level_continuation_adopt_retention_(keep);
  store.publish(std::move(prepared));
  level_continuation_install_retention_(std::move(keep));
}
typename continuation_authority_type::Base continuation_read() const {
  if (!continuation_parent_phase_)
    return level_continuation_read();
  auto& store = continuation_parent_working_;
  return continuation_local_phase(
      [&] {
        level_continuation_require_time_(store, store.read(), level_continuation_macro_start_);
        auto& live = finite_values_();
        live.finite.import_high_water(store.domain_high_water());
        live.transport.capture(store.read());
        return continuation_parent_authority_.read(store);
      },
      "AMR immutable global parent read");
}
continuation_ref continuation_working_value() const {
  return level_continuation_working_.at(active_level_).read();
}
double continuation_macro_start() const {
  return level_continuation_macro_start_;
}
double continuation_macro_end() const {
  return level_continuation_macro_end_;
}

continuation_ref continuation_child_reports(const FiniteDomainRef& domain) const {
  return continuation_local_phase(
      [&] {
        if (!continuation_parent_phase_ || !domain ||
            domain->size != static_cast<std::int64_t>(continuation_children_.size()))
          throw std::invalid_argument("AMR report domain does not cover the exact child forest");
        std::vector<continuation_ref> reports;
        reports.reserve(continuation_children_.size());
        for (std::size_t i = 0; i < continuation_children_.size(); ++i) {
          const auto& child = level_continuation_pending_.at(static_cast<int>(i));
          if (child.report != continuation_children_[i].value()->members()[2])
            throw std::logic_error("AMR report detached from its accepted child");
          reports.push_back(child.report);
        }
        return ContinuationValue<Dim>::array(domain, std::move(reports));
      },
      "AMR exact child observation array");
}

void continuation_publish(const typename continuation_authority_type::Accepted& accepted) const {
  if (!continuation_parent_phase_) {
    level_continuation_publish(accepted);
    return;
  }
  auto& root = continuation_parent_working_;
  level_continuation_selected_consensus_(continuation_parent_authority_, accepted);
  auto value = continuation_local_phase(
      [&] {
        continuation_parent_authority_.require_exact_children(accepted, continuation_children_);
        auto out = continuation_parent_authority_.require_root_publication(accepted, root);
        level_continuation_require_frontier_(root, out, level_continuation_macro_end_,
                                             static_cast<std::int64_t>(facade_->macro_step()) +
                                                 (continuation_regrid_phase() ? 0 : 1));
        // Accepted requires the exact dependent child graph to remain live.
        for (const auto& child : continuation_children_)
          (void)child.value();
        return out;
      },
      "AMR complete parent vote");
  auto root_image = level_continuation_prepare_(root, std::move(value), false);
  level_continuation_image_consensus_(root, root_image);
  auto prepared = continuation_local_phase(
      [&] {
        std::map<int, ContinuationStore<Dim>> stores;
        auto parent = root;
        auto parent_image = root_image;
        parent.publish(std::move(parent_image));
        stores.emplace(-1, std::move(parent));
        for (const auto& [level, pending] : level_continuation_pending_) {
          auto child = continuation_regrid_phase()
                           ? require_continuation_regrid_().targets.at(level)
                           : level_continuation_macro_bases_.at(level);
          auto image = pending.image;
          child.publish(std::move(image));
          stores.emplace(level, std::move(child));
        }
        return stores;
      },
      "AMR complete forest publication preparation");
  auto keep = continuation_prepare_forest_retention_(prepared, &amr_continuation_accepted_);
  // Every complete image and assessment precedes the first projection. The
  // enclosing native macro transaction restores all levels on allocation/copy/fence failure.
  for (const auto& [level, store] : prepared) {
    if (level < 0)
      continue;
    with_program_resource_level(level, [&] {
      auto projections = continuation_local_phase(
          [&] {
            std::vector<std::pair<field_type*, const field_type*>> fields;
            fields.reserve(store.configuration().fields.size());
            for (const auto& binding : store.configuration().fields) {
              const auto& field = continuation_at<Dim>(store.read(), binding.path)->field();
              fields.emplace_back(&state(binding.program_block), &field);
            }
            return fields;
          },
          "AMR accepted child field projection preparation");
      commit_many(std::span<const std::pair<field_type*, const field_type*>>(projections));
    });
  }
  amr_continuation_accepted_.swap(prepared);
  if (continuation_regrid_phase())
    require_continuation_regrid_().published = true;
  continuation_install_forest_retention_(std::move(keep));
  continuation_parent_authority_.revoke();
  continuation_parent_phase_ = false;
  finish_level_continuation_attempt();
  affine_finalization_.reset();
}

private:
mutable bool continuation_parent_phase_ = false;
mutable ContinuationStore<Dim> continuation_parent_working_;
mutable continuation_authority_type continuation_parent_authority_;
mutable std::vector<typename continuation_authority_type::Accepted> continuation_children_;
void continuation_require_forest_(const std::map<int, ContinuationStore<Dim>>& stores) const {
  const int levels = nlev();
  if (stores.size() != static_cast<std::size_t>(levels) + 1 || !stores.contains(-1))
    throw std::invalid_argument("AMR continuation forest does not cover its whole hierarchy");
  const auto& root = stores.at(-1);
  if (!root.initialized() || !root.configuration().fields.empty())
    throw std::invalid_argument("AMR continuation forest lacks its unique global parent");
  for (int level = 0; level < levels; ++level) {
    const auto& child = stores.at(level);
    if (!child.initialized() ||
        child.configuration().fields.size() != static_cast<std::size_t>(n_blocks()))
      throw std::invalid_argument("AMR continuation forest lacks a complete child");
    if (std::bit_cast<std::uint64_t>(
            continuation_at<Dim>(root.read(), root.configuration().time_path)->real()) !=
            std::bit_cast<std::uint64_t>(
                continuation_at<Dim>(child.read(), child.configuration().time_path)->real()) ||
        continuation_at<Dim>(root.read(), root.configuration().step_path)->integer() !=
            continuation_at<Dim>(child.read(), child.configuration().step_path)->integer())
      throw std::invalid_argument("AMR continuation forest frontiers are not synchronized");
  }
}
void continuation_revoke_attempt_() const noexcept {
  continuation_field_storage_.clear();
  end_continuation_regrid();
  continuation_discard_restart_levels_();
  // Snapshot/restart has already installed the exact restored retention plan.
  // Purging live capabilities must never replace that plan with a later trial's keep.
  finish_level_continuation_attempt();
  revoke_finite_values_on_rollback();
  level_continuation_rollback_keep_.clear();
  continuation_parent_authority_.revoke();
  continuation_children_.clear();
  continuation_parent_phase_ = false;
  affine_finalization_.reset();
}
