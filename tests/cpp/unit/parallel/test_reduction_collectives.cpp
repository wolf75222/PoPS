#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/consensus_words.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// The collectives of the Program reductions (spec 0.3 SV20, duties 2 and 3): the compensated
// transport of dot_all and of the AMR composite pairing and integral, and ConsensusWords. The
// operator and encoding cases need no MPI; the transport cases also run under mpiexec, as the
// CTest variants np2, np3 and np4.

namespace {

using pops::CollectiveCompensatedSum;
using pops::FiniteCompensatedSum;
using pops::Real;
using Words = std::array<Real, 2>;
namespace transport = pops::mf_arith_detail;

constexpr std::string_view kLaneIdentity = "test/reduction-collectives";
constexpr Real kMax = std::numeric_limits<Real>::max();
constexpr Real kNaN = std::numeric_limits<Real>::quiet_NaN();
constexpr const char* kNonfiniteLocal = "Program dot_all local summary is nonfinite";
constexpr const char* kNonfiniteSum = "Program dot_all collective sum is nonfinite";

FiniteCompensatedSum summary(std::initializer_list<Real> terms) {
  FiniteCompensatedSum result;
  for (const Real term : terms)
    result.add(term);
  return result;
}

FiniteCompensatedSum nonfinite_summary() {
  FiniteCompensatedSum result;
  result.invalid = true;
  return result;
}

bool same_bits(Real left, Real right) {
  return std::memcmp(&left, &right, sizeof(Real)) == 0;
}

bool same_bits(const Words& left, const Words& right) {
  return same_bits(left[0], right[0]) && same_bits(left[1], right[1]);
}

/// left o right, as the MPI operator evaluates it: `left` holds the lower ranks.
Words join(const Words& left, Words right) {
  transport::join_compensated_words(left.data(), right.data());
  return right;
}

std::vector<Words> rank_words(const std::vector<FiniteCompensatedSum>& summaries) {
  std::vector<Words> words;
  for (std::size_t rank = 0; rank < summaries.size(); ++rank)
    words.push_back(transport::compensated_summary_words(summaries[rank], false, rank == 0));
  return words;
}

/// The transport of b6: every summary merged in rank order into a running sum started at zero.
FiniteCompensatedSum sequential_merge(const std::vector<FiniteCompensatedSum>& summaries) {
  FiniteCompensatedSum merged;
  for (const FiniteCompensatedSum& rank_summary : summaries) {
    merged.add(rank_summary.high);
    merged.add(rank_summary.low);
  }
  return merged;
}

Words left_to_right(const std::vector<Words>& words) {
  Words reduced = words.front();
  for (std::size_t rank = 1; rank < words.size(); ++rank)
    reduced = join(reduced, words[rank]);
  return reduced;
}

/// Every bracketing of words[first, last) that keeps rank order: the evaluations that an MPI
/// library may choose for a non-commutative operator.
std::vector<Words> bracketings(const std::vector<Words>& words, std::size_t first,
                               std::size_t last) {
  if (last - first == 1)
    return {words[first]};
  std::vector<Words> results;
  for (std::size_t split = first + 1; split < last; ++split)
    for (const Words& left : bracketings(words, first, split))
      for (const Words& right : bracketings(words, split, last))
        results.push_back(join(left, right));
  return results;
}

Real reduced_value(const Words& words) {
  return transport::compensated_summary_result(words).value;
}

template <class Call>
std::string overflow_message(Call&& call) {
  try {
    (void)call();
  } catch (const std::overflow_error& error) {
    return error.what();
  } catch (const std::exception& error) {
    return std::string("unexpected exception: ") + error.what();
  }
  return "no overflow_error";
}

/// Random rank summaries whose words have mixed magnitudes, with cancellation across ranks in a
/// third of the trials.
FiniteCompensatedSum random_summary(std::mt19937_64& random, int trial, int rank) {
  std::uniform_real_distribution<double> exponent(-20.0, 20.0);
  FiniteCompensatedSum result;
  const int terms = 1 + (trial + rank) % 9;
  for (int term = 0; term < terms; ++term) {
    const Real magnitude = std::pow(Real{10}, static_cast<Real>(exponent(random)));
    result.add((random() & 1U) != 0U ? magnitude : -magnitude);
    if (trial % 3 == 0)
      result.add(rank % 2 == 0 ? Real{1e16} : Real{-1e16});
  }
  return result;
}

class CompensatedSummaryTransport : public ::testing::Test {
 protected:
  [[nodiscard]] pops::CommunicatorView communicator() const { return lane_.communicator(); }
  [[nodiscard]] int rank() const { return lane_.rank(); }
  [[nodiscard]] int size() const { return lane_.size(); }
  [[nodiscard]] const pops::ExecutionLane& lane() const { return lane_; }

  /// The words of every rank, in rank order.
  [[nodiscard]] std::vector<Words> gathered(const Words& local) const {
    std::vector<Words> all(static_cast<std::size_t>(size()), local);
#ifdef POPS_HAS_MPI
    if (communicator().active() && size() > 1) {
      const int status = MPI_Allgather(local.data(), 2, pops::mpi_real_datatype(), all.data(), 2,
                                       pops::mpi_real_datatype(), communicator().native_handle());
      if (status != MPI_SUCCESS)
        throw std::runtime_error("test MPI_Allgather failed");
    }
#endif
    return all;
  }

  /// Whether every rank holds these bits.
  [[nodiscard]] bool every_rank_holds(Real value) const {
    for (const Words& words : gathered(Words{value, Real{0}}))
      if (!same_bits(words[0], value))
        return false;
    return true;
  }

 private:
  pops::ExecutionLane lane_ = pops::ExecutionLane::world(kLaneIdentity);
};

}  // namespace

TEST(CompensatedSummaryJoin, JoinsTheLowerRanksFirstSoTheOperatorIsNotCommutative) {
  const FiniteCompensatedSum lower = summary({0.1, 0.2});
  const FiniteCompensatedSum upper = summary({0.1, 1e-17});
  const Words lower_words{lower.high, lower.low};
  const Words upper_words{upper.high, upper.low};

  FiniteCompensatedSum lower_then_upper = lower;
  lower_then_upper.add(upper.high);
  lower_then_upper.add(upper.low);
  const Words joined = join(lower_words, upper_words);
  EXPECT_TRUE(same_bits(joined, Words{lower_then_upper.high, lower_then_upper.low}));
  EXPECT_FALSE(same_bits(joined, join(upper_words, lower_words)))
      << "the operands are not interchangeable, so the operator must stay non-commutative";
}

TEST(CompensatedSummaryJoin, KeepsTheCauseOfHighestPrecedenceFromEitherOperand) {
  const Words valid{1.0, 0.0};
  const Words failure{kNaN, transport::kCompensatedLocalFailure};
  const Words nonfinite{kNaN, transport::kCompensatedLocalNonfinite};
  const Words overflow{kNaN, transport::kCompensatedMergeOverflow};
  const auto cause = [](const Words& words) {
    EXPECT_TRUE(std::isnan(words[0]));
    return words[1];
  };
  for (const Words& invalid : {failure, nonfinite, overflow}) {
    EXPECT_EQ(cause(join(valid, invalid)), invalid[1]);
    EXPECT_EQ(cause(join(invalid, valid)), invalid[1]);
  }
  EXPECT_EQ(cause(join(failure, nonfinite)), transport::kCompensatedLocalFailure);
  EXPECT_EQ(cause(join(nonfinite, failure)), transport::kCompensatedLocalFailure);
  EXPECT_EQ(cause(join(overflow, failure)), transport::kCompensatedLocalFailure);
  EXPECT_EQ(cause(join(nonfinite, overflow)), transport::kCompensatedLocalNonfinite);
  EXPECT_EQ(cause(join(overflow, nonfinite)), transport::kCompensatedLocalNonfinite);
  EXPECT_EQ(cause(join(Words{0.75 * kMax, 0.0}, Words{0.75 * kMax, 0.0})),
            transport::kCompensatedMergeOverflow);
}

TEST(CompensatedSummaryWords, SendALocalFailureBeforeANonfiniteSummary) {
  const FiniteCompensatedSum finite = summary({1.0, 1e-16, 1e-16});
  for (const bool first_rank : {true, false}) {
    const Words failed =
        transport::compensated_summary_words(nonfinite_summary(), true, first_rank);
    EXPECT_TRUE(std::isnan(failed[0]));
    EXPECT_EQ(failed[1], transport::kCompensatedLocalFailure);
    const Words failed_finite = transport::compensated_summary_words(finite, true, first_rank);
    EXPECT_TRUE(std::isnan(failed_finite[0]));
    EXPECT_EQ(failed_finite[1], transport::kCompensatedLocalFailure);
    const Words nonfinite =
        transport::compensated_summary_words(nonfinite_summary(), false, first_rank);
    EXPECT_TRUE(std::isnan(nonfinite[0]));
    EXPECT_EQ(nonfinite[1], transport::kCompensatedLocalNonfinite);
  }
  EXPECT_TRUE(same_bits(transport::compensated_summary_words(finite, false, false),
                        Words{finite.high, finite.low}));
}

TEST(CompensatedSummaryWords, ReadBackTheSumAFailureOrTheOverflowOfTheirCause) {
  const CollectiveCompensatedSum sum = transport::compensated_summary_result(Words{2.0, 0.25});
  EXPECT_FALSE(sum.failed);
  EXPECT_EQ(sum.value, 2.25);
  const CollectiveCompensatedSum failed =
      transport::compensated_summary_result(Words{kNaN, transport::kCompensatedLocalFailure});
  EXPECT_TRUE(failed.failed);
  EXPECT_EQ(failed.value, 0.0);
  EXPECT_EQ(overflow_message([] {
              return transport::compensated_summary_result(
                  Words{kNaN, transport::kCompensatedLocalNonfinite});
            }),
            kNonfiniteLocal);
  EXPECT_EQ(overflow_message([] {
              return transport::compensated_summary_result(
                  Words{kNaN, transport::kCompensatedMergeOverflow});
            }),
            kNonfiniteSum);
}

TEST(CompensatedSummaryJoin, RankZeroEntersAsTheRunningSumSoLeftToRightIsTheSequentialMerge) {
  // The low word of (1.0, 2e-16) is almost one ulp: joined as it is, the first join rounds
  // differently from the sequential merge, which starts from a running sum at zero.
  const std::vector<FiniteCompensatedSum> summaries{summary({1.0, 1e-16, 1e-16}),
                                                    summary({-1.0, 1e-17})};
  const Real sequential = sequential_merge(summaries).value();
  const std::vector<Words> words = rank_words(summaries);
  EXPECT_TRUE(same_bits(reduced_value(left_to_right(words)), sequential));
  const Words unnormalized_first{summaries[0].high, summaries[0].low};
  EXPECT_FALSE(same_bits(reduced_value(join(unnormalized_first, words[1])), sequential));

  std::mt19937_64 random(20261107U);
  for (int trial = 0; trial < 600; ++trial) {
    const int ranks = 2 + trial % 7;
    std::vector<FiniteCompensatedSum> trial_summaries;
    for (int trial_rank = 0; trial_rank < ranks; ++trial_rank)
      trial_summaries.push_back(random_summary(random, trial, trial_rank));
    const Real reduced = reduced_value(left_to_right(rank_words(trial_summaries)));
    ASSERT_TRUE(same_bits(reduced, sequential_merge(trial_summaries).value()))
        << "trial " << trial << " over " << ranks << " ranks";
  }
}

TEST(CompensatedSummaryJoin, AnotherBracketingCanRoundTheLastBitsDifferently) {
  const std::vector<FiniteCompensatedSum> summaries{summary({0.7, 1e-16}), summary({1e16, 0.7}),
                                                    summary({-1e16, 0.7})};
  const std::vector<Words> words = rank_words(summaries);
  const Real sequential = sequential_merge(summaries).value();
  const Real lower_first = reduced_value(join(join(words[0], words[1]), words[2]));
  const Real upper_first = reduced_value(join(words[0], join(words[1], words[2])));
  EXPECT_TRUE(same_bits(lower_first, sequential));
  EXPECT_FALSE(same_bits(upper_first, sequential));
  EXPECT_LE(std::abs(upper_first - sequential),
            2 * std::numeric_limits<Real>::epsilon() * std::abs(sequential));
}

TEST(CompensatedSummaryJoin, ABracketingWhosePartialJoinOverflowsRefusesTheSum) {
  // The sequential merge stays finite, as do the bracketings that join rank 2 after rank 1 or
  // rank 0; a tree that joins ranks 2 and 3 first overflows, and the sum is refused.
  const std::vector<FiniteCompensatedSum> summaries{summary({-0.375 * kMax}),
                                                    summary({-0.375 * kMax}),
                                                    summary({0.75 * kMax}), summary({0.75 * kMax})};
  const FiniteCompensatedSum sequential = sequential_merge(summaries);
  ASSERT_TRUE(sequential.finite());
  const std::vector<Words> words = rank_words(summaries);
  EXPECT_TRUE(same_bits(reduced_value(left_to_right(words)), sequential.value()));
  EXPECT_EQ(overflow_message([&] {
              return transport::compensated_summary_result(
                  join(join(words[0], words[1]), join(words[2], words[3])));
            }),
            kNonfiniteSum);
  int refused = 0;
  for (const Words& reduced : bracketings(words, 0, words.size()))
    refused += std::isnan(reduced[0]) ? 1 : 0;
  EXPECT_EQ(refused, 3) << "of the five bracketings that keep rank order";
}

TEST(ConsensusWords, RefusesWordsBeyondItsCapacityAndReadsBeforeItsReduction) {
  const pops::ExecutionLane lane = pops::ExecutionLane::world(kLaneIdentity);
  pops::ConsensusWords<3> words;
  const std::size_t failure = words.failure(false);
  const std::size_t agreement = words.agreement(5);
  EXPECT_THROW((void)words.maximum(1.0), std::logic_error);
  EXPECT_THROW((void)words.failed(failure), std::logic_error);
  EXPECT_THROW((void)words.differs(agreement), std::logic_error);
  words.reduce(lane);
  EXPECT_FALSE(words.failed(failure));
  EXPECT_FALSE(words.differs(agreement));
  EXPECT_EQ(words.value(agreement), 5.0);
  EXPECT_THROW((void)words.value(3), std::logic_error);
  EXPECT_THROW((void)words.failure(true), std::logic_error);
  EXPECT_THROW(words.reduce(lane), std::logic_error);

  pops::ConsensusWords<1> one_word;
  EXPECT_THROW((void)one_word.agreement(1), std::logic_error);
}

TEST_F(CompensatedSummaryTransport, MpiRouteInitializesTheRequestedCommunicator) {
  // The MPI CTest variants name their rank count; the serial registration runs on one rank.
  const char* expected_ranks = std::getenv("POPS_TEST_EXPECT_RANKS");
  ASSERT_EQ(size(), expected_ranks == nullptr ? 1 : std::atoi(expected_ranks))
      << "the CTest route must initialize the requested communicator";
}

TEST_F(CompensatedSummaryTransport, EveryRankReceivesTheSameBitsFromOneBracketingInRankOrder) {
  std::mt19937_64 random(1234567U + 7919U * static_cast<std::uint64_t>(rank()));
  for (int trial = 0; trial < 300; ++trial) {
    const FiniteCompensatedSum local = random_summary(random, trial, rank());
    const Real reduced = pops::collective_finite_compensated_sum(local, communicator());
    ASSERT_TRUE(every_rank_holds(reduced)) << "trial " << trial;

    std::vector<FiniteCompensatedSum> summaries;
    for (const Words& words : gathered(Words{local.high, local.low})) {
      FiniteCompensatedSum rank_summary;
      rank_summary.high = words[0];
      rank_summary.low = words[1];
      summaries.push_back(rank_summary);
    }
    const std::vector<Words> words = rank_words(summaries);
    bool matched = false;
    for (const Words& candidate : bracketings(words, 0, words.size()))
      matched = matched || same_bits(reduced_value(candidate), reduced);
    ASSERT_TRUE(matched) << "trial " << trial << ": the reduced sum is no bracketing in rank order";
    if (size() <= 2)
      ASSERT_TRUE(same_bits(reduced, sequential_merge(summaries).value()))
          << "trial " << trial << ": up to two ranks the sum is the sequential merge of b6";
  }
}

TEST_F(CompensatedSummaryTransport, TheRegisteredOperatorJoinsBothWordsInRankOrder) {
  // Both words, not only their sum: an operator registered as commutative may swap operands,
  // which changes the words even where their sum survives. The np3 CTest variant forces the
  // Open MPI algorithm that does swap them. Without MPI the single rank keeps its own words.
  std::mt19937_64 random(7654321U + 104729U * static_cast<std::uint64_t>(rank()));
  for (int trial = 0; trial < 300; ++trial) {
    const Words local = transport::compensated_summary_words(random_summary(random, trial, rank()),
                                                             false, rank() == 0);
    Words reduced = local;
#ifdef POPS_HAS_MPI
    const auto& reduction = transport::compensated_summary_reduction();
    if (communicator().active() && size() > 1)
      ASSERT_EQ(MPI_Allreduce(MPI_IN_PLACE, reduced.data(), 1, reduction.pair, reduction.join,
                              communicator().native_handle()),
                MPI_SUCCESS);
#endif
    for (const Words& held : gathered(reduced))
      ASSERT_TRUE(same_bits(held, reduced)) << "trial " << trial << ": ranks hold other words";
    const std::vector<Words> words = gathered(local);
    bool matched = false;
    for (const Words& candidate : bracketings(words, 0, words.size()))
      matched = matched || same_bits(candidate, reduced);
    ASSERT_TRUE(matched) << "trial " << trial << ": the words are no bracketing in rank order";
  }
}

TEST_F(CompensatedSummaryTransport, ACancellationSplitOverTheRanksIsExact) {
  FiniteCompensatedSum local;
  if (rank() == 0)
    local.add(1e16);
  if (rank() == (size() > 1 ? 1 : 0))
    local.add(1.0);
  if (rank() == size() - 1)
    local.add(-1e16);
  EXPECT_EQ(pops::collective_finite_compensated_sum(local, communicator()), 1.0);
}

TEST_F(CompensatedSummaryTransport, ALocalFailureOfTheLastRankReachesEveryRank) {
  const CollectiveCompensatedSum reduced = pops::collective_finite_compensated_sum_or_failure(
      summary({1.0}), rank() == size() - 1, communicator());
  EXPECT_TRUE(reduced.failed);
  EXPECT_EQ(reduced.value, 0.0);
}

TEST_F(CompensatedSummaryTransport, AFailureComesBeforeANonfiniteSummaryAndThatBeforeAnOverflow) {
  // A nonfinite summary on rank 0 and a failure on the last rank: every rank learns the failure.
  {
    const FiniteCompensatedSum local = rank() == 0 ? nonfinite_summary() : summary({1.0});
    const CollectiveCompensatedSum reduced = pops::collective_finite_compensated_sum_or_failure(
        local, size() == 1 || rank() == size() - 1, communicator());
    EXPECT_TRUE(reduced.failed);
  }
  // A nonfinite summary on the last rank while the others overflow when joined.
  {
    const FiniteCompensatedSum local =
        rank() == size() - 1 ? nonfinite_summary() : summary({0.75 * kMax});
    EXPECT_EQ(overflow_message(
                  [&] { return pops::collective_finite_compensated_sum(local, communicator()); }),
              kNonfiniteLocal);
  }
  // Finite summaries whose sum overflows.
  {
    const FiniteCompensatedSum local = summary({0.75 * kMax});
    if (size() == 1) {
      EXPECT_EQ(pops::collective_finite_compensated_sum(local, communicator()), 0.75 * kMax);
    } else {
      EXPECT_EQ(overflow_message(
                    [&] { return pops::collective_finite_compensated_sum(local, communicator()); }),
                kNonfiniteSum);
    }
  }
}

TEST_F(CompensatedSummaryTransport, ConsensusWordsDecideFailuresAgreementsAndMaximaTogether) {
  pops::ConsensusWords<8> words;
  const std::size_t no_failure = words.failure(false);
  const std::size_t last_rank_failure = words.failure(rank() == size() - 1);
  const std::size_t equal = words.agreement(7);
  const std::size_t unequal = words.agreement(rank() == 0 ? 3 : 4);
  const std::size_t largest = words.maximum(static_cast<double>(rank()) - 0.5);
  const std::size_t empty = words.maximum(-std::numeric_limits<double>::infinity());
  words.reduce(lane());
  EXPECT_FALSE(words.failed(no_failure));
  EXPECT_TRUE(words.failed(last_rank_failure));
  EXPECT_FALSE(words.differs(equal));
  EXPECT_EQ(words.differs(unequal), size() > 1);
  EXPECT_EQ(words.value(largest), static_cast<double>(size() - 1) - 0.5);
  EXPECT_EQ(words.value(empty), -std::numeric_limits<double>::infinity());
}
