#pragma once

/// @file
/// @brief Release-active validation errors for public/runtime contracts.

#include <stdexcept>
#include <string>
#include <utility>

namespace pops {

inline std::string validation_message(const std::string& where, const std::string& expected,
                                      const std::string& received) {
  return "pops validation error [" + where + "]: expected " + expected + "; received " + received;
}

class ValidationError : public std::runtime_error {
 public:
  ValidationError(std::string where, std::string expected, std::string received)
      : std::runtime_error(validation_message(where, expected, received)),
        where_(std::move(where)),
        expected_(std::move(expected)),
        received_(std::move(received)) {}

  const std::string& where() const noexcept { return where_; }
  const std::string& expected() const noexcept { return expected_; }
  const std::string& received() const noexcept { return received_; }

 private:
  std::string where_;
  std::string expected_;
  std::string received_;
};

[[noreturn]] inline void throw_validation_error(std::string where, std::string expected,
                                                std::string received) {
  throw ValidationError(std::move(where), std::move(expected), std::move(received));
}

inline void require_validation(bool condition, std::string where, std::string expected,
                               std::string received) {
  if (!condition)
    throw_validation_error(std::move(where), std::move(expected), std::move(received));
}

}  // namespace pops
