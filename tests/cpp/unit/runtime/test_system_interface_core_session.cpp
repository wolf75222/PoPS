#include <gtest/gtest.h>
#include <pops/runtime/system/system_block_store.hpp>
#include <pops/mesh/boundary/prepared_boundary_component.hpp>

#include <stdexcept>
#include <thread>

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
