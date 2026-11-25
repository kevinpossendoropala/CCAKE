// Runs the vector and matrix self-checks of include/unit_tests.h under GoogleTest.
//
// Those checks print one "PASSED" or "FAILED" line each and never fail the
// program. Here their output is captured and turned into a real test result:
// no line may say FAILED, and the number of PASSED lines must be the number of
// checks. The checks themselves are used exactly as they are.
#include <gtest/gtest.h>

#include <string>

#include "unit_tests.h"

namespace {

int count_occurrences(const std::string& text, const std::string& word)
{
  int count = 0;
  for (std::string::size_type pos = text.find(word); pos != std::string::npos;
       pos = text.find(word, pos + word.size()))
    ++count;
  return count;
}

} // namespace

TEST(LegacyChecks, Vector)
{
  testing::internal::CaptureStdout();
  run_vector_tests();
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(count_occurrences(output, "FAILED"), 0) << output;
  EXPECT_EQ(count_occurrences(output, "PASSED"), 12) << output;
}

TEST(LegacyChecks, Matrix)
{
  testing::internal::CaptureStdout();
  run_matrix_tests();
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(count_occurrences(output, "FAILED"), 0) << output;
  EXPECT_EQ(count_occurrences(output, "PASSED"), 21) << output;
}
