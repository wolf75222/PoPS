#include <gtest/gtest.h>
#include <pops/runtime/system/system_block_store.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>
#include <pops/mesh/boundary/prepared_boundary_component.hpp>

#include <stdexcept>
#include <set>
#include <thread>

namespace {
struct NativeFactoryProbe {
  pops::SystemBlockClosures<pops::kNativeDimension>::ExternalGhostBoundary ghost;
  std::function<void(const void*)> observe_transport;
};
thread_local NativeFactoryProbe* native_factory_probe = nullptr;
}  // namespace

namespace pops {
template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  auto prepared = prepare_generated_system_block(std::move(request));
  if constexpr (Dim == kNativeDimension) {
    if (native_factory_probe) {
      // Instrument the real generated hook and transport. The System issuer and all of its
      // collective ownership checks remain production code; external ABI is covered in Python.
      *prepared.closures.external_ghost_boundary = native_factory_probe->ghost;
      auto physical = prepared.closures.boundary_flux_full_at_point_prepared;
      auto observe = native_factory_probe->observe_transport;
      prepared.closures.boundary_flux_full_at_point_prepared =
          [physical, observe](const auto& point, auto& state, auto& output, const auto& boundary,
                              const auto& lane, const auto& transport) {
            observe(&transport);
            physical(point, state, output, boundary, lane, transport);
          };
    }
  }
  return prepared;
}
}  // namespace pops

namespace {
template <int Dim>
struct Fixture {
  using Store = pops::SystemBlockStore<Dim>;
  using Provider = pops::SystemInterfaceProvider<Dim>;
  using Session = typename Provider::CoreSession;
  using Field = pops::MultiFab<Dim>;
  Store store;
  Field state, output;
  typename Provider::point_type point{};
  int executions = 0;
  bool fail_core = false;

  explicit Fixture(
      typename Provider::Evaluate evaluate,
      typename Provider::CoreAdmission admission = [](void (*action)(void*), void* payload) {
        action(payload);
      }) {
    point.clock = "main";
    point.dt = 0.001;
    point.physical_time = 0.0;
    store.blocks.resize(1);
    Provider provider;
    provider.provider_identity = "exact-test-provider";
    provider.collective_contract = "exact-test-contract";
    provider.evaluate_rhs = evaluate;
    provider.evaluate_core = std::move(evaluate);
    provider.has_interfaces = [](int block) { return block == 0; };
    provider.evaluation_count = [](const auto&, int) { return std::size_t{0}; };
    provider.discard = [] {};
    store.install_interface_provider(
        std::move(provider),
        [this](const auto& received_point, const auto& states, const auto& outputs,
               const auto& modes) {
          EXPECT_EQ(received_point, point);
          ASSERT_EQ(states.size(), 1);
          ASSERT_EQ(outputs.size(), 1);
          ASSERT_EQ(modes.size(), 1);
          EXPECT_EQ(states[0], &state);
          EXPECT_EQ(outputs[0], &output);
          EXPECT_EQ(modes[0], 1);
          ++executions;
          if (fail_core)
            throw std::runtime_error("core evaluation failure");
        },
        std::move(admission));
  }

  void run(bool core = false) {
    if (core)
      store.evaluate_rhs_core_with_interfaces(point, {&state}, {&output}, {1});
    else
      store.evaluate_rhs_with_interfaces(point, {&state}, {&output}, {1});
  }
};

template <int Dim>
void lifetime_and_exact_group() {
  typename Fixture<Dim>::Session retained;
  Fixture<Dim> fixture(
      [&](const auto&, const auto&, const auto&, const auto&, const auto& session) {
        retained = session;
        session->evaluate();
        EXPECT_THROW(session->evaluate(), std::logic_error);
      });
  fixture.run();
  EXPECT_EQ(fixture.executions, 1);
  ASSERT_TRUE(retained);
  EXPECT_THROW(retained->evaluate(), std::logic_error);
  fixture.run(true);
  EXPECT_EQ(fixture.executions, 2);
  EXPECT_THROW(retained->evaluate(), std::logic_error);
  EXPECT_THROW(fixture.store.evaluate_rhs_with_interfaces(fixture.point, {}, {}, {}),
               std::invalid_argument);
  EXPECT_THROW(
      fixture.store.evaluate_rhs_with_interfaces(fixture.point, {&fixture.state}, {nullptr}, {1}),
      std::invalid_argument);
  EXPECT_EQ(fixture.executions, 2);
  EXPECT_THROW(fixture.store.evaluate_rhs_with_interfaces(fixture.point, {&fixture.state},
                                                          {&fixture.state}, {1}),
               std::invalid_argument);
  fixture.store.blocks.resize(2);
  typename Fixture<Dim>::Field second_state;
  EXPECT_THROW(
      fixture.store.evaluate_rhs_with_interfaces(fixture.point, {&fixture.state, &second_state},
                                                 {&fixture.output, &fixture.output}, {1, 1}),
      std::invalid_argument);
  EXPECT_THROW(
      fixture.store.evaluate_rhs_with_interfaces(fixture.point, {&fixture.state, &second_state},
                                                 {&second_state, &fixture.output}, {1, 1}),
      std::invalid_argument);
  EXPECT_EQ(fixture.executions, 2);
}

template <int Dim>
void cross_store_and_thread() {
  typename Fixture<Dim>::Session outer;
  Fixture<Dim> second([&](const auto&, const auto&, const auto&, const auto&, const auto& current) {
    ASSERT_TRUE(outer);
    EXPECT_THROW(outer->evaluate(), std::logic_error);
    current->evaluate();
  });
  Fixture<Dim> first([&](const auto&, const auto&, const auto&, const auto&, const auto& current) {
    outer = current;
    bool refused = false;
    std::thread foreign([&] {
      try {
        current->evaluate();
      } catch (const std::logic_error&) {
        refused = true;
      }
    });
    foreign.join();
    EXPECT_TRUE(refused);
    second.run();
    current->evaluate();
  });
  first.run();
  EXPECT_EQ(first.executions, 1);
  EXPECT_EQ(second.executions, 1);
  EXPECT_THROW(outer->evaluate(), std::logic_error);
}

template <int Dim>
void exceptions_and_missing_core() {
  typename Fixture<Dim>::Session retained;
  Fixture<Dim> before([&](const auto&, const auto&, const auto&, const auto&, const auto& current) {
    retained = current;
    throw std::runtime_error("provider failure before core");
  });
  EXPECT_THROW(before.run(), std::runtime_error);
  EXPECT_EQ(before.executions, 0);
  EXPECT_THROW(retained->evaluate(), std::logic_error);
  Fixture<Dim> after([&](const auto&, const auto&, const auto&, const auto&, const auto& current) {
    retained = current;
    current->evaluate();
    throw std::runtime_error("provider failure after core");
  });
  EXPECT_THROW(after.run(), std::runtime_error);
  EXPECT_EQ(after.executions, 1);
  EXPECT_THROW(retained->evaluate(), std::logic_error);
  Fixture<Dim> absent([](const auto&, const auto&, const auto&, const auto&, const auto&) {});
  EXPECT_THROW(absent.run(), std::logic_error);
  EXPECT_EQ(absent.executions, 0);
  Fixture<Dim> swallowed(
      [&](const auto&, const auto&, const auto&, const auto&, const auto& current) {
        retained = current;
        EXPECT_THROW(current->evaluate(), std::runtime_error);
        EXPECT_THROW(current->evaluate(), std::logic_error);
      });
  swallowed.fail_core = true;
  EXPECT_THROW(swallowed.run(), std::logic_error);
  EXPECT_EQ(swallowed.executions, 1);
  EXPECT_THROW(retained->evaluate(), std::logic_error);
}

TEST(SystemInterfaceCoreSession, admission_failure_is_collective_before_any_core_or_callback) {
  const auto lane = pops::ExecutionLane::world("tests.interface-core-admission");
  bool inject = true;
  int callbacks = 0;
  Fixture<2> fixture(
      [&](const auto&, const auto&, const auto&, const auto&, const auto& current) {
        ++callbacks;
        current->evaluate();
      },
      [&](void (*action)(void*), void* payload) {
        pops::runtime::program::collective_boundary_provider_phase(
            lane, "injected core admission", [&] {
              action(payload);
              if (inject && lane.rank() == lane.size() - 1)
                throw std::bad_alloc();
            });
      });
  EXPECT_ANY_THROW(fixture.run());
  EXPECT_EQ(callbacks, 0);
  EXPECT_EQ(fixture.executions, 0);
  inject = false;
  fixture.run();
  EXPECT_EQ(callbacks, 1);
  EXPECT_EQ(fixture.executions, 1);
}

TEST(SystemInterfaceCoreSession, exact_group_and_lifetime_all_dimensions) {
  lifetime_and_exact_group<1>();
  lifetime_and_exact_group<2>();
  lifetime_and_exact_group<3>();
}
TEST(SystemInterfaceCoreSession, two_owning_stores_and_foreign_thread_all_dimensions) {
  cross_store_and_thread<1>();
  cross_store_and_thread<2>();
  cross_store_and_thread<3>();
}
TEST(SystemInterfaceCoreSession, exception_and_missing_core_all_dimensions) {
  exceptions_and_missing_core<1>();
  exceptions_and_missing_core<2>();
  exceptions_and_missing_core<3>();
}
}  // namespace

namespace {
constexpr int kDim = pops::kNativeDimension;
using NativeSystem = pops::System<kDim>;
using NativeField = pops::MultiFab<kDim>;
using NativeProvider = pops::SystemInterfaceProvider<kDim>;
using NativePoint = NativeProvider::point_type;

struct NativeFixture {
  std::shared_ptr<pops::ExecutionLane> lane;
  std::unique_ptr<NativeSystem> system;
  std::unique_ptr<NativeField> left_output, right_output;
  NativeProvider::CoreSession retained;
  std::function<void()> before_core;
  int completed = 0;

  explicit NativeFixture(const std::string& identity, bool interfaces = true,
                         NativeFactoryProbe* probe = nullptr) {
    pops::SystemConfig<kDim> config;
    for (int axis = 0; axis < kDim; ++axis) {
      config.shape[axis] = 8;
      config.lower[axis] = 0;
      config.upper[axis] = 1;
      config.periodicity[axis] = probe == nullptr;
    }
    config.boxes = {pops::Box<kDim>::from_extents(config.shape)};
    system = std::make_unique<NativeSystem>(config);
    lane = std::make_shared<pops::ExecutionLane>(
        pops::ExecutionLane::duplicate_world_collectively(identity));
    system->install_prepared_boundary_execution_lane(lane);
    for (const auto* name : {"left", "right"})
      system->install_block_state_route(name, identity + "/" + name + "/state");
    system->seal_auxiliary_providers();
    pops::RealVector<kDim> velocity{};
    velocity[0] = pops::Real(0.25);
    struct ProbeScope {
      explicit ProbeScope(NativeFactoryProbe* selected) { native_factory_probe = selected; }
      ~ProbeScope() { native_factory_probe = nullptr; }
    } probe_scope(probe);
    for (const auto* name : {"left", "right"}) {
      pops::add_compiled_model(*system, name, pops::nd::ScalarAdvection<kDim>::prepare(velocity),
                               "none", "rusanov", "conservative", "explicit");
      if (probe) {
        std::vector<std::string> identities;
        for (int face = 0; face < 2 * kDim; ++face)
          identities.push_back(identity + "/" + name + "/face/" + std::to_string(face));
        auto boundary = std::make_shared<const pops::PreparedHyperbolicBoundary<kDim>>(
            pops::prepare_hyperbolic_boundary<kDim>(std::vector<std::string>(2 * kDim, "foextrap"),
                                                    std::vector<double>(2 * kDim, 0.0), identities,
                                                    {"Scalar"}));
        system->install_prepared_hyperbolic_boundary(name, identity + "/" + name + "/boundary", 1,
                                                     identity + "/" + name + "/state", boundary);
      }
    }
    auto allocate = [&](int block) {
      auto& state = system->block_state(block);
      state.set_val(pops::Real(block + 1));
      return std::make_unique<NativeField>(state.layout(), state.distribution(), state.local_rank(),
                                           state.ncomp(), state.ghosts());
    };
    left_output = allocate(0);
    right_output = allocate(1);
    NativeProvider provider;
    provider.provider_identity = identity + "/provider";
    provider.collective_contract = identity + "/contract";
    provider.has_interfaces = [](int block) { return block == 0 || block == 1; };
    provider.evaluation_count = [this](const auto&, int) { return std::size_t(completed); };
    provider.discard = [] {};
    provider.evaluate_rhs = provider.evaluate_core = [this](const auto&, const auto&, const auto&,
                                                            const auto&, const auto& core) {
      retained = core;
      if (before_core)
        before_core();
      core->evaluate();  // Intentionally no provider preflight: the real System owns it.
      ++completed;
    };
    if (interfaces)
      system->install_interface_provider(std::move(provider));
    // The installed issuer must follow the stable Impl, not the moved facade address.
    system = std::make_unique<NativeSystem>(std::move(*system));
  }

  NativePoint point() const {
    NativePoint result;
    result.clock = "macro";
    result.dt = 0.001;
    result.physical_time = 0.0;
    result.graph_identity = "exact-group-graph";
    result.rate_identity = "exact-group-rate";
    result.application_identity = "exact-group-application";
    return result;
  }

  void reset_outputs() {
    left_output->set_val(pops::Real(-17));
    right_output->set_val(pops::Real(-17));
  }

  void evaluate(const NativePoint& at, std::vector<int> blocks = {0, 1},
                std::vector<int> modes = {1, 1}, bool alias = false) {
    std::vector<NativeField*> states, outputs;
    for (const auto block : blocks) {
      states.push_back(&system->block_state(block));
      outputs.push_back(block == 0 ? left_output.get() : right_output.get());
    }
    if (alias)
      outputs[0] = &system->block_state(1);
    system->block_rhs_group(at, blocks, states, outputs, modes);
  }

  void expect_outputs(pops::Real value) const {
    for (auto* output : {left_output.get(), right_output.get()})
      for (std::size_t local = 0; local < output->local_size(); ++local) {
        auto host = output->fab(local).create_host_mirror();
        output->fab(local).copy_to_host(host);
        const auto valid = output->fab(local).box();
        const auto grown = output->fab(local).grown_box();
        for (std::int64_t cell = 0; cell < valid.numPts(); ++cell) {
          auto remaining = cell;
          std::size_t offset = 0, stride = 1;
          for (int axis = 0; axis < kDim; ++axis) {
            const auto index = valid.lo[axis] + remaining % valid.length(axis);
            remaining /= valid.length(axis);
            offset += static_cast<std::size_t>(index - grown.lo[axis]) * stride;
            stride *= static_cast<std::size_t>(grown.length(axis));
          }
          EXPECT_EQ(host(offset), value);
        }
      }
  }
};
}  // namespace

TEST(SystemInterfaceCoreSession, real_system_exact_request_after_move_and_stale_refusal) {
  NativeFixture fixture("interface-real-system");
  fixture.reset_outputs();
  fixture.evaluate(fixture.point());
  EXPECT_EQ(fixture.completed, 1);
  fixture.expect_outputs(pops::Real(0));
  EXPECT_THROW(fixture.retained->evaluate(), std::logic_error);
  fixture.reset_outputs();
  fixture.evaluate(fixture.point());
  EXPECT_EQ(fixture.completed, 2);
  fixture.expect_outputs(pops::Real(0));
}

TEST(SystemInterfaceCoreSession, real_system_rejects_divergent_group_before_output_then_retries) {
  if (pops::n_ranks() < 2)
    GTEST_SKIP() << "requires distinct MPI ranks";
  NativeFixture fixture("interface-real-consensus");
  const bool last_rank = pops::my_rank() == pops::n_ranks() - 1;
  const char* cases[] = {"point",         "clock", "graph",        "rate",          "application",
                         "active_blocks", "mode",  "invalid_mode", "invalid_point", "output_alias"};
  for (const auto* kind : cases) {
    SCOPED_TRACE(kind);
    auto point = fixture.point();
    std::vector<int> blocks{0, 1}, modes{1, 1};
    bool alias = false;
    if (last_rank) {
      const std::string selected(kind);
      if (selected == "point")
        point.physical_time = 0.125;
      if (selected == "clock")
        point.clock = "other-clock";
      if (selected == "graph")
        point.graph_identity = "other-graph";
      if (selected == "rate")
        point.rate_identity = "other-rate";
      if (selected == "application")
        point.application_identity = "other-application";
      if (selected == "active_blocks") {
        blocks.pop_back();
        modes.pop_back();
      }
      if (selected == "mode")
        modes[1] = 0;
      if (selected == "invalid_mode")
        modes[1] = 2;
      if (selected == "invalid_point")
        point.dt = -1;
      if (selected == "output_alias")
        alias = true;
    }
    fixture.reset_outputs();
    const auto completed = fixture.completed;
    EXPECT_THROW(fixture.evaluate(point, blocks, modes, alias), std::runtime_error);
    EXPECT_EQ(fixture.completed, completed);
    fixture.expect_outputs(pops::Real(-17));
    fixture.evaluate(fixture.point());
    EXPECT_EQ(fixture.completed, completed + 1);
    fixture.expect_outputs(pops::Real(0));
  }
}

TEST(SystemInterfaceCoreSession, real_system_cannot_consume_another_system_active_session) {
  NativeFixture outer("interface-real-outer"), inner("interface-real-inner");
  bool refused = false;
  inner.before_core = [&] {
    EXPECT_THROW(outer.retained->evaluate(), std::logic_error);
    refused = true;
  };
  outer.before_core = [&] { inner.evaluate(inner.point()); };
  outer.evaluate(outer.point());
  EXPECT_TRUE(refused);
  EXPECT_EQ(outer.completed, 1);
  EXPECT_EQ(inner.completed, 1);
  outer.expect_outputs(pops::Real(0));
  inner.expect_outputs(pops::Real(0));
  EXPECT_THROW(outer.retained->evaluate(), std::logic_error);
  EXPECT_THROW(inner.retained->evaluate(), std::logic_error);
}

TEST(SystemInterfaceCoreSession,
     real_physical_group_without_interfaces_retains_transport_and_ghost_hook) {
  int evaluations = 0;
  bool fail_once = true;
  std::set<const void*> transports;
  NativeFactoryProbe probe;
  probe.observe_transport = [&](const void* address) { transports.insert(address); };
  probe.ghost = [&](const auto&, auto&, const auto&, const auto& lane) {
    ++evaluations;
    const bool fail = std::exchange(fail_once, false);
    pops::runtime::program::collective_boundary_provider_phase(lane, "native ghost test", [&] {
      if (fail && lane.rank() == lane.size() - 1)
        throw std::runtime_error("injected external ghost failure");
    });
  };
  NativeFixture fixture("physical-group-without-interface", false, &probe);
  fixture.reset_outputs();
  EXPECT_THROW(fixture.evaluate(fixture.point()), std::runtime_error);
  EXPECT_EQ(evaluations, 0);
  fixture.expect_outputs(pops::Real(-17));
  fixture.system->mark_bound();
  EXPECT_THROW(fixture.evaluate(fixture.point()), std::runtime_error);
  EXPECT_EQ(evaluations, 1);
  fixture.expect_outputs(pops::Real(-17));
  fixture.evaluate(fixture.point());
  EXPECT_EQ(evaluations, 3);
  fixture.expect_outputs(pops::Real(0));
  ASSERT_EQ(transports.size(), 2);
  const auto prepared = transports;
  fixture.reset_outputs();
  fixture.evaluate(fixture.point());
  EXPECT_EQ(evaluations, 5);
  fixture.expect_outputs(pops::Real(0));
  EXPECT_EQ(transports, prepared);
}

TEST(SystemInterfaceCoreSession, real_system_rank_local_boundary_discard_retains_runtime_lane) {
  int evaluations = 0;
  NativeFactoryProbe probe;
  probe.observe_transport = [](const void*) {};
  probe.ghost = [&](const auto&, auto&, const auto&, const auto&) { ++evaluations; };
  NativeFixture fixture("physical-group-discard-lane", false, &probe);
  const auto retained = fixture.lane;
  const bool last_rank = retained->rank() == retained->size() - 1;
  if (last_rank)
    fixture.system->discard_hyperbolic_boundaries();
  ASSERT_EQ(&fixture.system->prepared_boundary_execution_lane(), retained.get());
  // Boundary transaction retry must retain the original lane; replacing it stays forbidden.
  EXPECT_THROW(fixture.system->install_prepared_boundary_execution_lane(retained), std::exception);
  fixture.reset_outputs();
  if (retained->size() > 1) {
    EXPECT_THROW(fixture.evaluate(fixture.point()), std::runtime_error);
    fixture.expect_outputs(pops::Real(-17));
  }
  EXPECT_EQ(evaluations, 0);
  fixture.system->discard_hyperbolic_boundaries();
  EXPECT_EQ(&fixture.system->prepared_boundary_execution_lane(), retained.get());
  fixture.evaluate(fixture.point());
  fixture.expect_outputs(pops::Real(0));
  EXPECT_EQ(evaluations, 0);
}
