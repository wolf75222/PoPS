#pragma once

// Deterministic CBOR identity values (RFC 8949, length-first deterministic ordering).
// The deliberately small vocabulary is shared with pops.identity: no floats, no opaque fallback.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::identity {

class CanonicalValue {
 public:
  enum class Kind { kNull, kBool, kInt, kText, kBytes, kArray, kMap, kSet };
  using Bytes = std::vector<std::uint8_t>;
  using Array = std::vector<CanonicalValue>;
  using Map = std::vector<std::pair<std::string, CanonicalValue>>;

  CanonicalValue() = default;
  explicit CanonicalValue(bool value) : kind_(Kind::kBool), bool_(value) {}
  explicit CanonicalValue(std::int64_t value) : kind_(Kind::kInt), int_(value) {}

  static CanonicalValue text(std::string value) {
    CanonicalValue out;
    out.kind_ = Kind::kText;
    out.text_ = std::move(value);
    return out;
  }
  static CanonicalValue bytes(Bytes value) {
    CanonicalValue out;
    out.kind_ = Kind::kBytes;
    out.bytes_ = std::move(value);
    return out;
  }
  static CanonicalValue array(Array value) {
    CanonicalValue out;
    out.kind_ = Kind::kArray;
    out.items_ = std::move(value);
    return out;
  }
  static CanonicalValue map(Map value) {
    CanonicalValue out;
    out.kind_ = Kind::kMap;
    out.map_ = std::move(value);
    return out;
  }
  static CanonicalValue set(Array value) {
    CanonicalValue out;
    out.kind_ = Kind::kSet;
    out.items_ = std::move(value);
    return out;
  }

  [[nodiscard]] Kind kind() const noexcept { return kind_; }
  [[nodiscard]] bool boolean() const noexcept { return bool_; }
  [[nodiscard]] std::int64_t integer() const noexcept { return int_; }
  [[nodiscard]] const std::string& text_value() const noexcept { return text_; }
  [[nodiscard]] const Bytes& bytes_value() const noexcept { return bytes_; }
  [[nodiscard]] const Array& items() const noexcept { return items_; }
  [[nodiscard]] const Map& mapping() const noexcept { return map_; }

 private:
  Kind kind_ = Kind::kNull;
  bool bool_ = false;
  std::int64_t int_ = 0;
  std::string text_;
  Bytes bytes_;
  Array items_;
  Map map_;
};

namespace detail {

inline bool valid_utf8(const std::string& value) {
  const auto* data = reinterpret_cast<const unsigned char*>(value.data());
  std::size_t i = 0;
  while (i < value.size()) {
    const unsigned char first = data[i++];
    if (first <= 0x7f)
      continue;
    int trailing = 0;
    std::uint32_t code = 0;
    std::uint32_t minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      trailing = 1;
      code = first & 0x1fU;
      minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      trailing = 2;
      code = first & 0x0fU;
      minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      trailing = 3;
      code = first & 0x07U;
      minimum = 0x10000;
    } else {
      return false;
    }
    if (i + static_cast<std::size_t>(trailing) > value.size())
      return false;
    for (int j = 0; j < trailing; ++j) {
      const unsigned char next = data[i++];
      if ((next & 0xc0U) != 0x80U)
        return false;
      code = (code << 6U) | (next & 0x3fU);
    }
    if (code < minimum || code > 0x10ffffU || (code >= 0xd800U && code <= 0xdfffU))
      return false;
  }
  return true;
}

inline void append_uint(CanonicalValue::Bytes& out, std::uint8_t major, std::uint64_t value) {
  const std::uint8_t prefix = static_cast<std::uint8_t>(major << 5U);
  if (value < 24U) {
    out.push_back(static_cast<std::uint8_t>(prefix | value));
  } else if (value <= 0xffU) {
    out.push_back(static_cast<std::uint8_t>(prefix | 24U));
    out.push_back(static_cast<std::uint8_t>(value));
  } else if (value <= 0xffffU) {
    out.push_back(static_cast<std::uint8_t>(prefix | 25U));
    out.push_back(static_cast<std::uint8_t>(value >> 8U));
    out.push_back(static_cast<std::uint8_t>(value));
  } else if (value <= 0xffffffffULL) {
    out.push_back(static_cast<std::uint8_t>(prefix | 26U));
    for (int shift = 24; shift >= 0; shift -= 8)
      out.push_back(static_cast<std::uint8_t>(value >> shift));
  } else {
    out.push_back(static_cast<std::uint8_t>(prefix | 27U));
    for (int shift = 56; shift >= 0; shift -= 8)
      out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

inline bool length_first_less(const CanonicalValue::Bytes& left,
                              const CanonicalValue::Bytes& right) {
  if (left.size() != right.size())
    return left.size() < right.size();
  return std::lexicographical_compare(left.begin(), left.end(), right.begin(), right.end());
}

inline void encode_into(const CanonicalValue& value, CanonicalValue::Bytes& out);

inline CanonicalValue::Bytes encoded(const CanonicalValue& value) {
  CanonicalValue::Bytes out;
  encode_into(value, out);
  return out;
}

inline void append_text(CanonicalValue::Bytes& out, const std::string& text) {
  if (!valid_utf8(text))
    throw std::invalid_argument("canonical identity text is not valid UTF-8");
  append_uint(out, 3, text.size());
  out.insert(out.end(), text.begin(), text.end());
}

inline void encode_into(const CanonicalValue& value, CanonicalValue::Bytes& out) {
  switch (value.kind()) {
    case CanonicalValue::Kind::kNull:
      out.push_back(0xf6);
      return;
    case CanonicalValue::Kind::kBool:
      out.push_back(value.boolean() ? 0xf5 : 0xf4);
      return;
    case CanonicalValue::Kind::kInt: {
      const std::int64_t integer = value.integer();
      if (integer >= 0) {
        append_uint(out, 0, static_cast<std::uint64_t>(integer));
      } else {
        // -(n + 1) is defined for INT64_MIN, whereas -n would overflow.
        append_uint(out, 1, static_cast<std::uint64_t>(-(integer + 1)));
      }
      return;
    }
    case CanonicalValue::Kind::kText:
      append_text(out, value.text_value());
      return;
    case CanonicalValue::Kind::kBytes:
      append_uint(out, 2, value.bytes_value().size());
      out.insert(out.end(), value.bytes_value().begin(), value.bytes_value().end());
      return;
    case CanonicalValue::Kind::kArray:
      append_uint(out, 4, value.items().size());
      for (const auto& item : value.items())
        encode_into(item, out);
      return;
    case CanonicalValue::Kind::kMap: {
      struct Entry {
        CanonicalValue::Bytes key;
        const CanonicalValue* value;
      };
      std::vector<Entry> entries;
      entries.reserve(value.mapping().size());
      for (const auto& pair : value.mapping()) {
        CanonicalValue::Bytes key;
        append_text(key, pair.first);
        entries.push_back({std::move(key), &pair.second});
      }
      std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return length_first_less(a.key, b.key);
      });
      for (std::size_t i = 1; i < entries.size(); ++i) {
        if (entries[i - 1].key == entries[i].key)
          throw std::invalid_argument("canonical identity map has a duplicate key");
      }
      append_uint(out, 5, entries.size());
      for (const auto& entry : entries) {
        out.insert(out.end(), entry.key.begin(), entry.key.end());
        encode_into(*entry.value, out);
      }
      return;
    }
    case CanonicalValue::Kind::kSet: {
      // RFC 8746 set tag 258, with deterministic order by canonical encoded item.
      append_uint(out, 6, 258);
      std::vector<CanonicalValue::Bytes> items;
      items.reserve(value.items().size());
      for (const auto& item : value.items())
        items.push_back(encoded(item));
      std::sort(items.begin(), items.end(), length_first_less);
      for (std::size_t i = 1; i < items.size(); ++i) {
        if (items[i - 1] == items[i])
          throw std::invalid_argument("canonical identity set has duplicate canonical elements");
      }
      append_uint(out, 4, items.size());
      for (const auto& item : items)
        out.insert(out.end(), item.begin(), item.end());
      return;
    }
  }
  throw std::logic_error("unknown canonical identity value kind");
}

}  // namespace detail

inline CanonicalValue::Bytes canonical_bytes(const CanonicalValue& value) {
  return detail::encoded(value);
}

}  // namespace pops::identity
