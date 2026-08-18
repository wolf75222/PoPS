#include <pops/runtime/native_package_finalize.hpp>

#include <gtest/gtest.h>

#include <exception>
#include <stdexcept>
#include <string>

TEST(NativePackageFinalizeInnerException, RethrowsInnerFailureAfterCollectiveRollback) {
  const auto failure = std::make_exception_ptr(
      std::runtime_error("native package bind/fft disagreed on rank 0"));
  try {
    pops::rethrow_native_package_finalize_failure(failure);
    FAIL() << "expected inner finalize exception";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "native package bind/fft disagreed on rank 0");
    EXPECT_STRNE(error.what(), pops::kNativePackageFinalizeCollectiveMessage);
  }
}

TEST(NativePackageFinalizeInnerException, GenericMessageOnlyWhenRankHasNoInnerFailure) {
  try {
    pops::rethrow_native_package_finalize_failure(std::exception_ptr{});
    FAIL() << "expected collective finalize exception";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), pops::kNativePackageFinalizeCollectiveMessage);
  }
}
