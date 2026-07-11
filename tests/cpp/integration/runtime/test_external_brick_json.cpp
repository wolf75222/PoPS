// Manifest JSON escape/unescape round-trip for the external-brick registry (Spec 3 section 21-22,
// ADC-463). The host emits each brick's id/category/requirements/capabilities into the JSON manifest
// `pops.lib.load_cpp_library` parses with `json.loads`; the C++ reader (`field`) parses it back. Both
// directions must agree on EVERY byte a user-chosen id/requirement can carry -- structural `"` / `\`
// and any control character -- or a manifest with such a token is silently truncated (C++ side) or
// rejected outright (`json.loads` raises on a raw control char). This is a pure-string test: it does
// NOT need Kokkos or _pops, so it stays a fast, always-on gate independent of the device toolchain.

#include <gtest/gtest.h>

#include <pops/runtime/program/external_brick.hpp>

#include <string>

using pops::runtime::program::BrickManifestEntry;
using pops::runtime::program::BrickRegistry;
using pops::runtime::program::json_escape;
using pops::runtime::program::json_unescape;
using pops::runtime::program::kBrickManifestSchemaVersion;

namespace {

// json_escape then json_unescape recovers the original byte-for-byte.
void expect_roundtrip(const std::string& raw, const char* label) {
  EXPECT_TRUE(json_unescape(json_escape(raw)) == raw) << label;
}

// A valid JSON string body carries no raw control char and no bare `"`; every backslash starts a
// recognized escape. This is the property `json.loads` needs (a raw control char makes it raise).
bool is_valid_json_string_body(const std::string& e) {
  for (std::size_t i = 0; i < e.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(e[i]);
    if (c < 0x20)
      return false;  // raw control char -> json.loads rejects
    if (c == '"')
      return false;  // bare quote -> ends the string early
    if (c == '\\') {
      if (i + 1 >= e.size())
        return false;
      ++i;  // skip the escaped char (json_escape only emits \" \\ \n \r \t \b \f \uXXXX)
    }
  }
  return true;
}

}  // namespace

TEST(ExternalBrickJson, RoundtripsOverTokenBytes) {
  // round-trips over the bytes a brick token can carry
  expect_roundtrip("my_riemann", "plain identifier");
  expect_roundtrip("B_z,T_e,rho", "requirements CSV (commas untouched)");
  expect_roundtrip("a\"b", "embedded double quote");
  expect_roundtrip("a\\b", "embedded backslash");
  expect_roundtrip("c:/p\\q", "windows-ish path");
  expect_roundtrip(std::string("line1\nline2\tcol\r"), "newline + tab + carriage return");
  expect_roundtrip(std::string("x\x01y\x1fz", 5), "low control bytes 0x01 / 0x1f");
  expect_roundtrip("", "empty string");
}

TEST(ExternalBrickJson, EscapedFormIsValidJsonStringBody) {
  // the escaped form is always a valid JSON string body (what json.loads requires)
  EXPECT_TRUE(is_valid_json_string_body(json_escape("a\"b\\c")))
      << "escaped quote+backslash is valid JSON";
  EXPECT_TRUE(is_valid_json_string_body(json_escape(std::string("n\nt\tctrl\x02", 8))))
      << "escaped control chars are valid JSON";

  // a value carrying an escaped quote is NOT truncated at the inner quote (the field-scan contract)
  EXPECT_TRUE(json_unescape(json_escape("pre\"post")) == "pre\"post")
      << "escaped quote not truncated";
}

// ADC-657: to_json emits schema_version 3 and all ten per-entry fields (native_id /
// supported_layouts / supported_platforms / params / options / exported_symbols) the host parser
// (pops.descriptors.parse_brick_manifest) accepts. Emitter + parser stay in lockstep on this field set.
TEST(ExternalBrickJson, ToJsonEmitsV3SchemaAndFields) {
  EXPECT_TRUE(kBrickManifestSchemaVersion == 3) << "schema version is v3 (ADC-657)";

  // A clean registry so the emitted JSON contains exactly the one entry we register here (this TU's
  // executable has no static POPS_REGISTER_BRICK, so the singleton starts empty; clear() is belt-and-
  // braces isolation from the sibling test below in the same binary).
  BrickRegistry& reg = BrickRegistry::instance();
  reg.clear();
  reg.register_brick({"json_brick", "riemann", "pressure,wave_speeds", "physical_flux",
                      "json_native", "uniform,amr", "cpu,mpi", "cs2", "reconstruct",
                      "pops_brick_residual"});
  const std::string out = reg.to_json();

  EXPECT_TRUE(out.find("\"schema_version\":3") != std::string::npos) << "stamps v3";

  // Each v2 field is emitted with its registered value.
  EXPECT_TRUE(out.find("\"native_id\":\"json_native\"") != std::string::npos) << "native_id";
  EXPECT_TRUE(out.find("\"supported_layouts\":\"uniform,amr\"") != std::string::npos) << "layouts";
  EXPECT_TRUE(out.find("\"supported_platforms\":\"cpu,mpi\"") != std::string::npos) << "platforms";
  EXPECT_TRUE(out.find("\"params\":\"cs2\"") != std::string::npos) << "params";
  EXPECT_TRUE(out.find("\"options\":\"reconstruct\"") != std::string::npos) << "options";
  EXPECT_TRUE(out.find("\"exported_symbols\":\"pops_brick_residual\"") != std::string::npos)
      << "exported_symbols";

  // The four required fields are still present alongside the v2 additions.
  EXPECT_TRUE(out.find("\"id\":\"json_brick\"") != std::string::npos) << "id";
  EXPECT_TRUE(out.find("\"category\":\"riemann\"") != std::string::npos) << "category";
  EXPECT_TRUE(out.find("\"requirements\":\"pressure,wave_speeds\"") != std::string::npos)
      << "requirements";
  EXPECT_TRUE(out.find("\"capabilities\":\"physical_flux\"") != std::string::npos) << "capabilities";
}

// A minimal schema-v3 row still supplies its native identity explicitly; documentary CSV fields may
// be empty but the loader never invents an id.
TEST(ExternalBrickJson, ToJsonEmitsExplicitIdentityForMinimalBrick) {
  BrickRegistry& reg = BrickRegistry::instance();
  reg.clear();
  reg.register_brick({"minimal", "riemann", "", "", "minimal"});
  const std::string out = reg.to_json();
  EXPECT_TRUE(out.find("\"native_id\":\"minimal\"") != std::string::npos)
      << "explicit native_id emitted";
  EXPECT_TRUE(out.find("\"supported_layouts\":\"\"") != std::string::npos) << "empty layouts emitted";
  EXPECT_TRUE(out.find("\"exported_symbols\":\"\"") != std::string::npos) << "empty symbols emitted";
}

TEST(ExternalBrickJson, DuplicateIdIsIdempotentOnlyForIdenticalRows) {
  BrickRegistry& reg = BrickRegistry::instance();
  reg.clear();
  const BrickManifestEntry row{"same", "riemann", "pressure", "physical_flux", "same_native",
                               "uniform", "cpu", "", "", "pops_same"};
  EXPECT_NO_THROW(reg.register_brick(row));
  EXPECT_NO_THROW(reg.register_brick(row));
  EXPECT_TRUE(reg.size() == 1);
  BrickManifestEntry conflict = row;
  conflict.native_id = "different_native";
  EXPECT_THROW(reg.register_brick(conflict), std::runtime_error);
  EXPECT_TRUE(reg.size() == 1);
  EXPECT_THROW(reg.register_brick({"", "riemann", "", "", "native"}), std::runtime_error);
  EXPECT_THROW(reg.register_brick({"missing_native", "riemann", "", ""}), std::runtime_error);
}
