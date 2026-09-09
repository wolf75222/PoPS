#include <gtest/gtest.h>

#include <pops/runtime/system/native_package_capability.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using pops::runtime::system::AuxiliaryComponentKey;
using pops::runtime::system::NativeEllipticAttachmentRole;
using pops::runtime::system::exact_native_system_package_contract;
using pops::runtime::system::require_native_elliptic_output_contract;
using pops::runtime::system::validate_native_elliptic_attachment_contract;
using Attachment = pops::runtime::system::PreparedNativeEllipticAttachment<2>;
using Package = pops::runtime::system::PreparedNativeSystemPackage<2>;

std::vector<AuxiliaryComponentKey> field_outputs() {
  return {{"case:plasma/field:electrostatic", "field", "potential", "phi"},
          {"case:plasma/field:electrostatic", "field", "electric", "x"},
          {"case:plasma/field:electrostatic", "field", "electric", "y"}};
}

Attachment output_attachment() {
  // Retain the five-field aggregate contract used by existing output-bearing packages.
  return {"electrostatic", "model:plasma/field_rhs", field_outputs(), -1,
          [](const pops::MultiFab<2>&, pops::MultiFab<2>&) {}};
}

Attachment rhs_attachment(std::string species = "electron") {
  Attachment result;
  result.field = "electrostatic";
  result.rhs_identity = "model:" + species + "/field_rhs";
  result.rhs = [](const pops::MultiFab<2>&, pops::MultiFab<2>&) {};
  result.role = NativeEllipticAttachmentRole::rhs_only;
  result.field_slot = "case:plasma/field:electrostatic";
  result.binding_identity = "case:plasma/block:" + species + "/rhs-binding";
  return result;
}

void expect_invalid(const Attachment& attachment) {
  EXPECT_THROW(validate_native_elliptic_attachment_contract(attachment), std::invalid_argument);
  EXPECT_THROW(require_native_elliptic_output_contract(attachment, field_outputs(), -1),
               std::invalid_argument);
}

TEST(NativeFieldAttachment, ExistingAggregateRetainsOutputAuthority) {
  const Attachment attachment = output_attachment();
  EXPECT_EQ(attachment.role, NativeEllipticAttachmentRole::output_and_rhs);
  EXPECT_TRUE(attachment.field_slot.empty());
  EXPECT_TRUE(attachment.binding_identity.empty());
  EXPECT_NO_THROW(validate_native_elliptic_attachment_contract(attachment));
  EXPECT_NO_THROW(require_native_elliptic_output_contract(attachment, field_outputs(), -1));
}

TEST(NativeFieldAttachment, DistinctRhsContributionsShareNegativeGradientOutputAuthority) {
  const Attachment electron = rhs_attachment("electron");
  const Attachment ion = rhs_attachment("ion");
  EXPECT_EQ(electron.field, ion.field);
  EXPECT_EQ(electron.field_slot, ion.field_slot);
  EXPECT_NE(electron.rhs_identity, ion.rhs_identity);
  EXPECT_NE(electron.binding_identity, ion.binding_identity);
  for (const Attachment* attachment : {&electron, &ion}) {
    SCOPED_TRACE(attachment->rhs_identity);
    EXPECT_TRUE(attachment->outputs.empty());
    EXPECT_EQ(attachment->gradient_sign, 1);
    EXPECT_NO_THROW(validate_native_elliptic_attachment_contract(*attachment));
    // The Case field owns these keys and -grad(phi); neither density contribution claims them.
    EXPECT_NO_THROW(require_native_elliptic_output_contract(*attachment, field_outputs(), -1));
  }
}

TEST(NativeFieldAttachment, BothRolesRequireFieldRhsIdentityCallbackAndValidSign) {
  for (const Attachment& valid : {output_attachment(), rhs_attachment()}) {
    SCOPED_TRACE(static_cast<int>(valid.role));
    auto invalid = valid;
    invalid.field.clear();
    expect_invalid(invalid);
    invalid = valid;
    invalid.rhs_identity.clear();
    expect_invalid(invalid);
    invalid = valid;
    invalid.rhs = {};
    expect_invalid(invalid);
    for (int sign : {0, -2, 2}) {
      invalid = valid;
      invalid.gradient_sign = sign;
      expect_invalid(invalid);
    }
  }
}

TEST(NativeFieldAttachment, RhsOnlyRequiresBothExactBindingIdentities) {
  auto attachment = rhs_attachment();
  attachment.field_slot.clear();
  expect_invalid(attachment);
  attachment = rhs_attachment();
  attachment.binding_identity.clear();
  expect_invalid(attachment);
}

TEST(NativeFieldAttachment, RhsOnlyCannotClaimOutputsOrGradientOrientation) {
  auto attachment = rhs_attachment();
  attachment.outputs = field_outputs();
  expect_invalid(attachment);
  attachment = rhs_attachment();
  attachment.gradient_sign = -1;
  expect_invalid(attachment);
}

TEST(NativeFieldAttachment, RoleRelabelingCannotBypassEitherContract) {
  auto attachment = rhs_attachment();
  attachment.role = NativeEllipticAttachmentRole::output_and_rhs;
  expect_invalid(attachment);
  attachment = output_attachment();
  attachment.role = NativeEllipticAttachmentRole::rhs_only;
  attachment.field_slot = "case:plasma/field:electrostatic";
  attachment.binding_identity = "case:plasma/block:electron/rhs-binding";
  attachment.gradient_sign = 1;
  expect_invalid(attachment);
  attachment = rhs_attachment();
  attachment.role = static_cast<NativeEllipticAttachmentRole>(std::uint8_t{255});
  expect_invalid(attachment);
}

TEST(NativeFieldAttachment, OutputAuthorityRejectsRhsRoleMetadata) {
  auto attachment = output_attachment();
  attachment.field_slot = "case:plasma/field:electrostatic";
  expect_invalid(attachment);
  attachment = output_attachment();
  attachment.binding_identity = "case:plasma/block:electron/rhs-binding";
  expect_invalid(attachment);
}

TEST(NativeFieldAttachment, OnlyExactLegacyFieldAllowsAnEmptyOutputClaim) {
  auto attachment = output_attachment();
  attachment.outputs.clear();
  expect_invalid(attachment);
  attachment.field = "fields_from_state";
  EXPECT_NO_THROW(validate_native_elliptic_attachment_contract(attachment));
  EXPECT_NO_THROW(require_native_elliptic_output_contract(attachment, field_outputs(), -1));
  EXPECT_THROW(require_native_elliptic_output_contract(attachment, field_outputs(), 1),
               std::logic_error);
  attachment.field = "fields_from_state_extra";
  expect_invalid(attachment);
}

TEST(NativeFieldAttachment, OutputAuthorityRequiresExactKeysAndGradientSign) {
  const Attachment attachment = output_attachment();
  EXPECT_THROW(require_native_elliptic_output_contract(attachment, field_outputs(), 1),
               std::logic_error);
  for (int changed_part = 0; changed_part != 7; ++changed_part) {
    SCOPED_TRACE(changed_part);
    auto expected = field_outputs();
    switch (changed_part) {
      case 0:
        expected[0].owner_qid = "case:another/field:electrostatic";
        break;
      case 1:
        expected[0].space_kind = "auxiliary";
        break;
      case 2:
        expected[0].space_name = "another_potential";
        break;
      case 3:
        expected[0].component = "another_phi";
        break;
      case 4:
        std::swap(expected[0], expected[1]);
        break;
      case 5:
        expected.pop_back();
        break;
      case 6:
        expected.push_back(expected.back());
        break;
    }
    EXPECT_THROW(require_native_elliptic_output_contract(attachment, expected, -1),
                 std::logic_error);
  }
}

TEST(NativeFieldAttachment, ExactPackageIdentitySealsRoleSlotAndBinding) {
  Package package{};
  package.consumer_qid = "case:plasma/block:electron";
  package.elliptic_attachments = {rhs_attachment()};
  const std::string original = exact_native_system_package_contract(package);
  EXPECT_EQ(original, exact_native_system_package_contract(package));
  // Fingerprinting must detect even a malformed relabeling before installation validation.
  auto changed = package;
  changed.elliptic_attachments[0].role = NativeEllipticAttachmentRole::output_and_rhs;
  EXPECT_NE(original, exact_native_system_package_contract(changed));
  changed = package;
  changed.elliptic_attachments[0].field_slot = "case:plasma/field:another";
  EXPECT_NE(original, exact_native_system_package_contract(changed));
  changed = package;
  changed.elliptic_attachments[0].binding_identity = "case:plasma/block:ion/rhs-binding";
  EXPECT_NE(original, exact_native_system_package_contract(changed));
}

}  // namespace
