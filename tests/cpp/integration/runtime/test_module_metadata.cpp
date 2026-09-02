// Locks the host-owned candidate-table metadata adapter
// (include/pops/runtime/program/module_metadata.hpp, Spec 2 / ADC-442). OperatorId is the
// registration index; the loader materializes these records before candidate preparation.
#include <gtest/gtest.h>

#include <pops/runtime/program/module_metadata.hpp>

#include <string>
#include <string_view>
#include <vector>

using namespace pops::runtime::program;

namespace {

ProgramAbiView abi_view(std::string_view value) {
  return {value.data(), static_cast<std::uint64_t>(value.size())};
}

ProgramModuleRecord module_row(std::string_view owner, std::string_view identity,
                               std::string_view kind = "module") {
  return {abi_view(identity), abi_view(kind), abi_view("signature"), abi_view("{}"),
          abi_view(owner)};
}

ProgramAbiTable module_table(const ProgramModuleRecord* rows, std::size_t count) {
  return {rows, static_cast<std::uint64_t>(count), sizeof(ProgramModuleRecord)};
}

}  // namespace

TEST(ModuleMetadata, DefaultDescriptorIsEmpty) {
  ModuleMetadata empty;
  EXPECT_TRUE(empty.operators.empty()) << "default ModuleMetadata has no operators";
  EXPECT_TRUE(empty.find("anything") == nullptr) << "find on an empty descriptor returns nullptr";
}

TEST(ModuleMetadata, HandBuiltDescriptorResolvesOperatorsByName) {
  ModuleMetadata m;
  m.operators.push_back(
      {0U, "model/a", "fields_from_state", "field_operator", "(U) -> Fields", "{}"});
  m.operators.push_back({1U, "model/a", "explicit_rhs", "local_rate", "(U, Fields) -> Rate(U)",
                         "{\"kind\":\"local_rate\"}"});
  m.state_spaces.push_back("U");
  m.field_spaces.push_back("fields");
  EXPECT_TRUE(m.find("explicit_rhs") != nullptr) << "find resolves a known operator";
  EXPECT_TRUE(m.find("model/a", "explicit_rhs") != nullptr)
      << "owner-qualified find resolves the exact operator";
  EXPECT_TRUE(m.find("explicit_rhs")->id == 1U) << "OperatorId is the registration index";
  EXPECT_TRUE(m.find("explicit_rhs")->kind == "local_rate") << "operator kind is carried";
  EXPECT_TRUE(m.find("nope") == nullptr) << "find on an unknown operator returns nullptr";
}

TEST(ModuleMetadata, CandidateTablesRejectMalformedModuleRecords) {
  ProgramInstallationTables tables;
  tables.module_operators.push_back({"rhs", "local_rate", "", "not-json", "model/a"});
  EXPECT_THROW((void)read_module_metadata(tables), std::runtime_error)
      << "the host rejects malformed copied module records before candidate preparation";
}

TEST(ModuleMetadata, CandidateTablesKeepRepeatedNamesSeparatedByOwner) {
  const ProgramModuleRecord operators[] = {
      module_row("model/a", "rhs", "local_rate"),
      module_row("model/b", "rhs", "local_rate"),
  };
  const ProgramModuleRecord state_spaces[] = {
      module_row("model/a", "U", "state"),
      module_row("model/b", "U", "state"),
  };
  const ProgramModuleRecord field_spaces[] = {
      module_row("model/a", "fields", "field"),
      module_row("model/b", "fields", "field"),
  };
  ProgramCandidateDescriptor descriptor{};
  descriptor.module_operators = module_table(operators, std::size(operators));
  descriptor.module_state_spaces = module_table(state_spaces, std::size(state_spaces));
  descriptor.module_field_spaces = module_table(field_spaces, std::size(field_spaces));

  std::size_t aggregate = 0;
  const ProgramInstallationTables tables =
      ProgramInstallationTables::materialize(descriptor, aggregate);
  const ModuleMetadata metadata = read_module_metadata(tables);
  ASSERT_EQ(metadata.operators.size(), 2U);
  EXPECT_NE(metadata.find("model/a", "rhs"), nullptr);
  EXPECT_NE(metadata.find("model/b", "rhs"), nullptr);
  EXPECT_EQ(metadata.find("rhs"), nullptr) << "unqualified lookup remains ambiguous";
  EXPECT_EQ(metadata.state_spaces, (std::vector<std::string>{"U", "U"}));
  EXPECT_EQ(metadata.state_space_owners, (std::vector<std::string>{"model/a", "model/b"}));
  EXPECT_EQ(metadata.field_spaces, (std::vector<std::string>{"fields", "fields"}));
  EXPECT_EQ(metadata.field_space_owners, (std::vector<std::string>{"model/a", "model/b"}));

  const auto expect_same_owner_duplicate = [&](ProgramAbiTable ProgramCandidateDescriptor::* table,
                                               const ProgramModuleRecord* rows) {
    ProgramCandidateDescriptor duplicate = descriptor;
    duplicate.*table = module_table(rows, 2);
    std::size_t duplicate_aggregate = 0;
    EXPECT_THROW((void)ProgramInstallationTables::materialize(duplicate, duplicate_aggregate),
                 std::invalid_argument);
  };
  const ProgramModuleRecord duplicate_operators[] = {operators[0], operators[0]};
  const ProgramModuleRecord duplicate_state_spaces[] = {state_spaces[0], state_spaces[0]};
  const ProgramModuleRecord duplicate_field_spaces[] = {field_spaces[0], field_spaces[0]};
  expect_same_owner_duplicate(&ProgramCandidateDescriptor::module_operators, duplicate_operators);
  expect_same_owner_duplicate(&ProgramCandidateDescriptor::module_state_spaces,
                              duplicate_state_spaces);
  expect_same_owner_duplicate(&ProgramCandidateDescriptor::module_field_spaces,
                              duplicate_field_spaces);
}

TEST(ModuleMetadata, CheckpointMetadataDoesNotForgePrimaryClockWithoutHistory) {
  ProgramInstallationTables tables;
  const ProgramCheckpointMetadata metadata = read_program_checkpoint_metadata(tables);
  EXPECT_TRUE(metadata.histories.empty());
  EXPECT_TRUE(metadata.logical_clock_identities.empty())
      << "the detached prepared schedule is the only primary-clock authority";
  EXPECT_EQ(metadata.temporal_provider_identity, kGlobalTemporalPartitionProvider);
}

TEST(ModuleMetadata, RequiredAuxParsesAuxArray) {
  // required_aux parses the "aux" array of an operator's requirements JSON (ADC-446, the
  // install-time validation input). A flat, closed vocabulary, scanned without a JSON library.
  EXPECT_TRUE(required_aux("{\"kind\":\"local_source\",\"aux\":[\"grad_x\",\"grad_y\"]}") ==
              std::vector<std::string>({"grad_x", "grad_y"}))
      << "required_aux extracts a two-name aux array";
  EXPECT_TRUE(required_aux("{\"kind\":\"local_linear_operator\",\"aux\":[\"B_z\"]}") ==
              std::vector<std::string>({"B_z"}))
      << "required_aux extracts a single-name aux array";
  EXPECT_TRUE(required_aux("{\"kind\":\"local_rate\"}").empty())
      << "required_aux on requirements without an aux key is empty";
  EXPECT_TRUE(required_aux("{\"kind\":\"field_operator\",\"aux\":[]}").empty())
      << "required_aux on an empty aux array is empty";
  EXPECT_TRUE(required_aux("").empty()) << "required_aux on an empty string is empty";
}

TEST(ModuleMetadata, RequiredSolverAnchorsKeyMatch) {
  // required_solver reads the scalar "solver" requirement (Spec criterion 24, ADC-466), and the
  // KEY match is anchored: an aux field literally named "solver", or any value equal to "solver",
  // must NOT be misread as a solver requirement (else a valid install is wrongly rejected).
  EXPECT_TRUE(required_solver("{\"kind\":\"field_operator\",\"solver\":\"geometric_mg\"}") ==
              "geometric_mg")
      << "required_solver extracts the solver requirement";
  EXPECT_TRUE(required_solver("{\"kind\":\"local_rate\"}").empty())
      << "required_solver without a solver key is empty";
  EXPECT_TRUE(required_solver("{\"aux\":[\"solver\"],\"foo\":\"bar\"}").empty())
      << "required_solver does not misread an aux field named 'solver' (anchored key match)";
  EXPECT_TRUE(required_solver("{\"aux\":[\"solver\"],\"kind\":\"field_operator\"}").empty())
      << "required_solver does not read the 'kind' value when 'solver' is only an aux element";
  EXPECT_TRUE(required_blocks("{\"kind\":\"local_source\",\"block\":[\"ions\"]}") ==
              std::vector<std::string>{"ions"})
      << "required_blocks extracts the block-instance requirement";
  EXPECT_TRUE(required_blocks("{\"kind\":\"field_operator\",\"aux\":[\"B_z\"]}").empty())
      << "required_blocks without a block key is empty";
}
