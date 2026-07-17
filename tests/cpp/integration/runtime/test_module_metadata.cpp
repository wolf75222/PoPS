// Locks the GeneratedModule metadata reader (include/pops/runtime/program/module_metadata.hpp,
// Spec 2 / ADC-442): the typed operator registry a problem.so exports for introspection and
// install-time validation. OperatorId is the registration index; incomplete metadata is rejected
// before installation. The loader-symbol integration test exercises a real shared object.
#include <gtest/gtest.h>

#include <pops/runtime/program/module_metadata.hpp>

#include <string>
#include <vector>

using namespace pops::runtime::program;

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

TEST(ModuleMetadata, MissingModuleContractIsRejected) {
  EXPECT_THROW((void)read_module_metadata(pops::dynlib::handle{}), std::runtime_error)
      << "a null module handle is invalid";
#if defined(_WIN32)
  pops::dynlib::handle self = ::GetModuleHandleW(nullptr);
#else
  pops::dynlib::handle self = ::dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
#endif
  ASSERT_TRUE(pops::dynlib::valid(self));
  EXPECT_THROW((void)read_module_metadata(self), std::runtime_error)
      << "a module without the complete metadata family is rejected";
#if !defined(_WIN32)
  pops::dynlib::close(self);
#endif
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
