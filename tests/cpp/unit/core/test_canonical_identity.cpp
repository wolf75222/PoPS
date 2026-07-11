#include <gtest/gtest.h>

#include <pops/core/identity/canonical_value.hpp>
#include <pops/core/identity/sha256.hpp>

#include <cstdint>
#include <string>

namespace {

using pops::identity::CanonicalValue;

std::string hex(const CanonicalValue& value) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string out;
  for (std::uint8_t byte : pops::identity::canonical_bytes(value)) {
    out.push_back(digits[byte >> 4U]);
    out.push_back(digits[byte & 0xfU]);
  }
  return out;
}

}  // namespace

TEST(CanonicalIdentity, EncodesRfc8949PreferredScalars) {
  EXPECT_EQ(hex(CanonicalValue()), "f6");
  EXPECT_EQ(hex(CanonicalValue(false)), "f4");
  EXPECT_EQ(hex(CanonicalValue(true)), "f5");
  EXPECT_EQ(hex(CanonicalValue(std::int64_t{23})), "17");
  EXPECT_EQ(hex(CanonicalValue(std::int64_t{24})), "1818");
  EXPECT_EQ(hex(CanonicalValue(std::int64_t{-24})), "37");
  EXPECT_EQ(hex(CanonicalValue(std::int64_t{-25})), "3818");
  EXPECT_EQ(hex(CanonicalValue(INT64_MAX)), "1b7fffffffffffffff");
  EXPECT_EQ(hex(CanonicalValue(INT64_MIN)), "3b7fffffffffffffff");
  EXPECT_EQ(hex(CanonicalValue::bytes({0x50, 0x6f, 0x50, 0x53})), "44506f5053");
  EXPECT_EQ(hex(CanonicalValue::text("\xc3\xa9")), "62c3a9");
}

TEST(CanonicalIdentity, OrdersMapsAndSetsByEncodedLengthThenBytes) {
  const auto mapping = CanonicalValue::map({
      {"\xc3\xa9", CanonicalValue(std::int64_t{3})},
      {"aa", CanonicalValue(std::int64_t{2})},
      {"b", CanonicalValue(std::int64_t{1})},
  });
  EXPECT_EQ(hex(mapping), "a36162016261610262c3a903");

  const auto set = CanonicalValue::set({CanonicalValue(std::int64_t{3}),
                                        CanonicalValue(std::int64_t{1}),
                                        CanonicalValue(std::int64_t{2})});
  EXPECT_EQ(hex(set), "d9010283010203");
}

TEST(CanonicalIdentity, RejectsInvalidTextAndDuplicateCanonicalEntries) {
  EXPECT_THROW(pops::identity::canonical_bytes(CanonicalValue::text("\xc0\x80")),
               std::invalid_argument);
  EXPECT_THROW(pops::identity::canonical_bytes(CanonicalValue::map(
                   {{"same", CanonicalValue()}, {"same", CanonicalValue(true)}})),
               std::invalid_argument);
  EXPECT_THROW(pops::identity::canonical_bytes(CanonicalValue::set(
                   {CanonicalValue(std::int64_t{1}), CanonicalValue(std::int64_t{1})})),
               std::invalid_argument);
}

TEST(CanonicalIdentity, Sha256MatchesFipsVectorsAndCanonicalPayload) {
  EXPECT_EQ(pops::identity::sha256_hex({}),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(pops::identity::sha256_hex({0x61, 0x62, 0x63}),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const auto mapping = CanonicalValue::map({
      {"\xc3\xa9", CanonicalValue(std::int64_t{3})},
      {"aa", CanonicalValue(std::int64_t{2})},
      {"b", CanonicalValue(std::int64_t{1})},
  });
  EXPECT_EQ(pops::identity::sha256_hex(pops::identity::canonical_bytes(mapping)),
            "897b081f52e7fa5b7aad2b8b0b5a3e4781a8bafc572b8967c847fc4533a27d7d");
}
