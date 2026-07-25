#pragma once

/// @file
/// @brief Typed prepared-callable protocol with exact communicator-comparable contracts.

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops {

namespace detail {

template <class Unsigned>
  requires(std::is_unsigned_v<Unsigned>)
inline void append_contract_unsigned(std::string& bytes, Unsigned value) {
  for (int shift = static_cast<int>(sizeof(Unsigned) * 8u) - 8; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>((value >> shift) & Unsigned{0xff}));
}

template <class Value>
concept ExactContractScalar =
    std::is_enum_v<std::remove_cv_t<Value>> || std::is_integral_v<std::remove_cv_t<Value>> ||
    std::same_as<std::remove_cv_t<Value>, float> || std::same_as<std::remove_cv_t<Value>, double>;

template <ExactContractScalar Value>
inline void append_contract_scalar_payload(std::string& bytes, Value value) {
  using T = std::remove_cv_t<Value>;
  if constexpr (std::is_enum_v<T>) {
    bytes.push_back('e');
    append_contract_scalar_payload(bytes, static_cast<std::underlying_type_t<T>>(value));
  } else if constexpr (std::same_as<T, bool>) {
    bytes.push_back('b');
    bytes.push_back(value ? char{1} : char{0});
  } else if constexpr (std::is_integral_v<T>) {
    bytes.push_back(std::is_signed_v<T> ? 'i' : 'u');
    bytes.push_back(static_cast<char>(sizeof(T)));
    append_contract_unsigned(bytes, static_cast<std::make_unsigned_t<T>>(value));
  } else if constexpr (std::same_as<T, float>) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    bytes.push_back('f');
    append_contract_unsigned(bytes, std::bit_cast<std::uint32_t>(value));
  } else {
    static_assert(std::same_as<T, double> && sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    bytes.push_back('d');
    append_contract_unsigned(bytes, std::bit_cast<std::uint64_t>(value));
  }
}

inline void append_contract_frame(std::string& destination, char kind, std::string_view payload) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  destination.push_back(kind);
  append_contract_unsigned(destination, static_cast<std::uint64_t>(payload.size()));
  destination.append(payload.data(), payload.size());
}

}  // namespace detail

/// Builds a byte-exact, canonical-byte-order semantic contract.
///
/// Every value is framed as ``kind | uint64-big-endian byte-count | payload``. Scalars additionally
/// encode signedness and width; floating-point values use their IEEE object bits. Text and opaque
/// bytes have distinct kinds. A sequence records both its element count and an individual frame for
/// every encoded element, so neither concatenation nor nesting can introduce an ambiguity. Use
/// fixed-width integer types when a contract must be identical across different C++ ABIs.
class ExactContractBuilder {
 public:
  ExactContractBuilder() = default;

  template <detail::ExactContractScalar Value>
  ExactContractBuilder& scalar(Value value) {
    std::string payload;
    detail::append_contract_scalar_payload(payload, value);
    append_frame_('s', payload);
    return *this;
  }

  ExactContractBuilder& text(std::string_view value) {
    append_frame_('t', value);
    return *this;
  }

  ExactContractBuilder& bytes(std::string_view value) {
    append_frame_('x', value);
    return *this;
  }

  ExactContractBuilder& bytes(std::span<const std::byte> value) {
    const auto* data = value.empty() ? "" : reinterpret_cast<const char*>(value.data());
    append_frame_('x', std::string_view(data, value.size()));
    return *this;
  }

  ExactContractBuilder& presence(bool present) {
    const char value = present ? char{1} : char{0};
    append_frame_('p', std::string_view(&value, 1));
    return *this;
  }

  template <class Provider>
    requires requires(const Provider& provider) {
      { static_cast<bool>(provider) } -> std::same_as<bool>;
      { provider.collective_contract() } -> std::convertible_to<std::string_view>;
    }
  ExactContractBuilder& optional_collective_contract(const Provider& provider) {
    const bool present = static_cast<bool>(provider);
    presence(present);
    if (present)
      bytes(provider.collective_contract());
    return *this;
  }

  template <std::ranges::input_range Range, class Encoder>
    requires requires(Encoder& encoder, ExactContractBuilder& element,
                      std::ranges::range_reference_t<Range> value) {
      { std::invoke(encoder, element, value) } -> std::same_as<void>;
    }
  ExactContractBuilder& sequence(Range&& values, Encoder encoder) {
    std::string elements;
    std::uint64_t count = 0;
    for (auto&& value : values) {
      if (count == std::numeric_limits<std::uint64_t>::max())
        throw std::length_error("exact contract sequence has too many elements");
      ExactContractBuilder element;
      std::invoke(encoder, element, value);
      detail::append_contract_frame(elements, 'v', element.view());
      ++count;
    }

    std::string payload;
    detail::append_contract_unsigned(payload, count);
    payload.append(elements);
    append_frame_('q', payload);
    return *this;
  }

  template <std::ranges::input_range Range>
    requires detail::ExactContractScalar<std::ranges::range_value_t<Range>>
  ExactContractBuilder& sequence(Range&& values) {
    using Value = std::ranges::range_value_t<Range>;
    return sequence(std::forward<Range>(values), [](ExactContractBuilder& element, auto&& value) {
      element.scalar(static_cast<Value>(value));
    });
  }

  [[nodiscard]] std::string_view view() const noexcept { return bytes_; }
  [[nodiscard]] const std::string& str() const noexcept { return bytes_; }
  [[nodiscard]] std::string release() && noexcept { return std::move(bytes_); }

 private:
  void append_frame_(char kind, std::string_view payload) {
    detail::append_contract_frame(bytes_, kind, payload);
  }

  std::string bytes_;
};

/// Compatibility convenience for scalar-only parameter lists.  New provider sources should write
/// their complete parameter structure directly to an ExactContractBuilder.
template <class... Values>
  requires((detail::ExactContractScalar<Values>) && ...)
inline std::string exact_provider_parameters(const Values&... values) {
  ExactContractBuilder contract;
  (contract.scalar(values), ...);
  return std::move(contract).release();
}

/// Stable semantic identity owned by a concrete provider source type.
struct PreparedProviderIdentity {
  std::string_view name;
  std::uint64_t version = 0;
};

/// Allocation-free compatibility decision owned by a prepared provider.
///
/// Zero is the protocol-wide success code and therefore carries no diagnostic. Every non-zero code
/// and its stable static reason belong to the concrete provider; consumers authenticate and report
/// them but never branch on their meaning. This lets third-party providers explain rejected request
/// shapes without extending a core enum or teaching the registry their identity.
struct PreparedProviderSupport {
  std::uint32_t code = 0;
  std::string_view reason{};

  [[nodiscard]] constexpr bool accepted() const noexcept { return code == 0; }
  [[nodiscard]] constexpr bool well_formed() const noexcept {
    return accepted() ? reason.empty() : !reason.empty();
  }
  [[nodiscard]] static constexpr PreparedProviderSupport accept() noexcept { return {}; }
  [[nodiscard]] static constexpr PreparedProviderSupport reject(
      std::uint32_t code, std::string_view reason) noexcept {
    return {code, reason};
  }
};

/// Canonical bytes for communicator authentication of a provider-owned support decision.
inline std::string exact_prepared_provider_support(const PreparedProviderSupport& support) {
  if (!support.well_formed())
    throw std::invalid_argument("prepared provider returned a malformed support decision");
  ExactContractBuilder contract;
  contract.text("pops.prepared-provider-support")
      .scalar(std::uint32_t{1})
      .scalar(support.code)
      .text(support.reason);
  return std::move(contract).release();
}

/// Extension protocol accepted by PreparedProvider.  A source type owns its identity and version,
/// serializes every resolved parameter, and supplies the callable implementation.  This prevents an
/// unrelated caller from pairing an arbitrary lambda with somebody else's contract by accident. It
/// makes semantic ownership reviewable; it cannot prove the behavior of arbitrary user code.
template <class Source, class Result, class... Args>
concept PreparedProviderSourceFor =
    std::copy_constructible<std::remove_cvref_t<Source>> &&
    std::invocable<const std::remove_cvref_t<Source>&, Args...> &&
    std::same_as<std::invoke_result_t<const std::remove_cvref_t<Source>&, Args...>, Result> &&
    requires(const std::remove_cvref_t<Source>& source, ExactContractBuilder& contract) {
      {
        std::remove_cvref_t<Source>::provider_identity()
      } noexcept -> std::same_as<PreparedProviderIdentity>;
      { source.serialize_exact_parameters(contract) } -> std::same_as<void>;
    };

template <class Signature>
class PreparedProvider;

/// Immutable type-erased prepared provider.  Normal construction accepts only a typed source that
/// satisfies PreparedProviderSourceFor.  The explicitly named trusted_extension() escape hatch is
/// reserved for ABI/plugin bridges whose source type cannot cross the boundary; consumers never
/// dispatch on provider identity.
template <class Result, class... Args>
class PreparedProvider<Result(Args...)> {
 public:
  using Function = std::function<Result(Args...)>;

  PreparedProvider() = default;
  PreparedProvider(const PreparedProvider&) = default;
  PreparedProvider(PreparedProvider&&) noexcept = default;
  PreparedProvider& operator=(const PreparedProvider&) = default;
  PreparedProvider& operator=(PreparedProvider&&) noexcept = default;

  template <class Source>
    requires(!std::same_as<std::remove_cvref_t<Source>, PreparedProvider> &&
             PreparedProviderSourceFor<Source, Result, Args...>)
  explicit PreparedProvider(Source source) {
    using S = std::remove_cvref_t<Source>;
    ExactContractBuilder parameters;
    source.serialize_exact_parameters(parameters);
    initialize_(S::provider_identity(), std::move(parameters).release(),
                Function([source = std::move(source)](Args... args) -> Result {
                  return std::invoke(source, std::forward<Args>(args)...);
                }));
  }

  /// Explicit trust boundary for loaders or binary plugins that cannot expose a concrete C++ source
  /// type. exact_parameters must be produced by ExactContractBuilder and must describe every value
  /// that can alter the callable's result. This is an explicit trust boundary: PoPS can compare the
  /// declared contract across consumers, but cannot prove an opaque callable honors it. Prefer typed
  /// source construction everywhere else.
  [[nodiscard]] static PreparedProvider trusted_extension(PreparedProviderIdentity identity,
                                                          std::string exact_parameters,
                                                          Function function) {
    PreparedProvider provider;
    provider.initialize_(identity, std::move(exact_parameters), std::move(function));
    return provider;
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(function_); }

  [[nodiscard]] const std::string& implementation() const noexcept { return implementation_; }
  [[nodiscard]] std::uint64_t implementation_version() const noexcept { return version_; }
  [[nodiscard]] std::string_view exact_parameters() const noexcept { return exact_parameters_; }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  Result operator()(Args... args) const {
    if (!function_)
      throw std::logic_error("cannot invoke an empty prepared provider");
    return std::invoke(function_, std::forward<Args>(args)...);
  }

 private:
  void initialize_(PreparedProviderIdentity identity, std::string exact_parameters,
                   Function function) {
    if (identity.name.empty())
      throw std::invalid_argument("prepared provider implementation identity must not be empty");
    if (identity.version == 0)
      throw std::invalid_argument("prepared provider implementation version must be positive");
    if (!function)
      throw std::invalid_argument("prepared provider callable must not be empty");

    implementation_ = std::string(identity.name);
    version_ = identity.version;
    exact_parameters_ = std::move(exact_parameters);
    function_ = std::move(function);

    ExactContractBuilder contract;
    contract.text("pops.prepared-provider")
        .scalar(std::uint32_t{2})
        .text(implementation_)
        .scalar(version_);
    contract.bytes(exact_parameters_);
    collective_contract_ = std::move(contract).release();
  }

  std::string implementation_;
  std::uint64_t version_ = 0;
  std::string exact_parameters_;
  Function function_;
  std::string collective_contract_;
};

}  // namespace pops
