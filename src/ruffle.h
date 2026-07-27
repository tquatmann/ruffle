#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <boost/math/distributions/beta.hpp>

#include "storm/adapters/IntervalAdapter.h"
#include "storm/models/sparse/Model.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/sparse/ModelComponents.h"
#include "storm/utility/builder.h"
#include "storm/utility/constants.h"

/*!
 * This header only depends on Storm's core model/matrix types, not on its logging, exception, or settings
 * machinery, so it can be dropped into other projects on its own. All output goes through RUFFLE_LOG;
 * define RUFFLE_SILENT before including this header to compile out that output entirely.
 */
#ifndef RUFFLE_SILENT
#define RUFFLE_LOG(message) (std::cout << message)
#else
#define RUFFLE_LOG(message) \
    do {                    \
    } while (false)
#endif

/*!
 * Transforms a concrete Markov model (DTMC/MDP) into a related model:
 *  - an interval model (IMDP) learned by sampling the input as a black-box system and computing
 *    Clopper-Pearson confidence intervals on the observed transition probabilities, or built
 *    deterministically by widening each concrete probability into an interval of a fixed width;
 *  - or a perturbed point model obtained by resampling the real distribution of each choice.
 * See the free functions at the end of this namespace for the entry points.
 */
namespace ruffle {

/*!
 * Implementation of ruffle's model transformations for a fixed input value type. Not meant to be used
 * directly; see the free functions at the end of this namespace instead.
 */
template<typename ValueType>
class LearningHelper {
   public:
    static inline ValueType const zero = storm::utility::zero<ValueType>();
    static inline ValueType const one = storm::utility::one<ValueType>();
    static inline ValueType const two = one + one;
    /// The interval type produced for this ValueType: storm::Interval for double, storm::RationalInterval otherwise.
    using Interval = std::conditional_t<std::is_same_v<ValueType, double>, storm::Interval, storm::RationalInterval>;
    static_assert(std::is_same_v<storm::IntervalBaseType<Interval>, ValueType>);

    /*!
     * @param distributionFailureProbability The failure probability lambda budgeted for a whole choice.
     * @param numberOfSuccessors Number of successors among which to distribute that budget.
     * @return The per-successor Clopper-Pearson failure probability, i.e. lambda divided evenly among successors.
     */
    static ValueType getComponentFailureProbability(ValueType distributionFailureProbability, uint64_t numberOfSuccessors) {
        if (numberOfSuccessors == 0) {
            throw std::invalid_argument("Tried to learn a choice without successors.");
        }
        return distributionFailureProbability / storm::utility::convertNumber<ValueType>(numberOfSuccessors);
    }

    /*!
     * @param k Number of samples that landed on this successor.
     * @param n Total number of samples drawn for the choice.
     * @param componentFailureProbability Per-successor failure probability, see getComponentFailureProbability.
     * @return The exact Clopper-Pearson confidence interval for this successor's true probability.
     */
    static Interval getClopperPearsonInterval(uint64_t k, uint64_t n, ValueType componentFailureProbability) {
        if (n < 1) {
            throw std::invalid_argument("Tried to obtain Clopper-Pearson interval for state-action pair with no samples.");
        }
        if (k > n) {
            throw std::invalid_argument("Tried to obtain Clopper-Pearson interval for state-action pair with more successes than samples. How?");
        }
        // The incomplete beta function inverse is an iterative, inherently floating-point computation, so it
        // cannot be instantiated for exact types like RationalNumber; do this part in double and convert back.
        double const halfFailProb = storm::utility::convertNumber<double>(componentFailureProbability) / 2.0;
        double const lowerBound = k > 0 ? boost::math::ibeta_inv(k, n - k + 1, halfFailProb) : 0.0;
        double const upperBound = k < n ? boost::math::ibetac_inv(k + 1, n - k, halfFailProb) : 1.0;

        return Interval(storm::utility::convertNumber<ValueType>(lowerBound), storm::utility::convertNumber<ValueType>(upperBound));
    }

    /*!
     * Interval estimate for one successor: the trivial [1, 1] interval if the choice is known to have only one
     * possible successor, otherwise the Clopper-Pearson interval obtained from the observed samples.
     */
    static Interval estimateSuccessorInterval(bool hasSingleSuccessor, uint64_t successorSamples, uint64_t totalSamples,
                                              ValueType componentFailureProbability) {
        return hasSingleSuccessor ? Interval(one, one) : getClopperPearsonInterval(successorSamples, totalSamples, componentFailureProbability);
    }

    /*!
     * @param lowerSum Sum of the lower bounds of a choice's (not necessarily normalized) successor intervals.
     * @param upperSum Sum of the corresponding upper bounds.
     * @return The feasible L1 diameter of the choice's successor confidence polytope.
     */
    static ValueType feasibleL1Diameter(ValueType const& lowerSum, ValueType const& upperSum) {
        ValueType const lowerSlack = one - lowerSum;
        ValueType const upperSlack = upperSum - one;
        ValueType const feasibleMass = std::max(std::min(lowerSlack, upperSlack), zero);
        return two * feasibleMass;
    }

    /*!
     * Widens a probability p that is strictly between 0 and 1 into an interval of the given width, centered
     * at p and clamped to stay within [0, 1]. Probabilities of exactly 0 or 1 are left as point intervals.
     */
    static Interval widenProbability(ValueType const& p, ValueType const& delta) {
        if (storm::utility::isZero(p) || storm::utility::isOne(p)) {
            return Interval(p, p);
        }
        ValueType const halfDelta = delta / two;
        ValueType const lowerCandidate = p - halfDelta;
        ValueType const upperCandidate = p + halfDelta;
        return Interval(std::max(zero, lowerCandidate), std::min(one, upperCandidate));
    }

    /*!
     * If `epsilon` is set, raises `value` (a lower interval bound) to at least min(p, epsilon), where p is
     * the real probability of the same transition. This can never push the value above p, so it can't
     * invalidate an interval (whose lower bound must stay <= p).
     */
    static ValueType applyEpsilonFloor(ValueType const& value, ValueType const& p, std::optional<ValueType> const& epsilon) {
        if (!epsilon.has_value()) {
            return value;
        }
        return std::max(value, std::min(p, epsilon.value()));
    }

    /*!
     * The successor distribution of a choice, represented as a CDF: successor i is reached with cumulative
     * probability `cumulative[i]` (strictly ascending; the last entry is the total probability mass).
     * Storing the CDF as a sorted vector lets sample() do a binary search instead of an O(n) linear rescan
     * of the whole distribution on every draw. Sampling is always done in double precision regardless of
     * ValueType, and explicit zero-probability entries are dropped.
     */
    struct SuccessorCdf {
        std::vector<double> cumulative;
        std::vector<uint64_t> successors;

        std::size_t size() const {
            return successors.size();
        }

        uint64_t successor(std::size_t i) const {
            return successors[i];
        }

        /// The probability mass of successor i, recovered from adjacent CDF entries.
        double probability(std::size_t i) const {
            return cumulative[i] - (i > 0 ? cumulative[i - 1] : 0.0);
        }

        /// Draws the successor whose CDF interval contains `quantile` (a uniform sample in [0, 1)).
        uint64_t sample(double quantile) const {
            auto it = std::upper_bound(cumulative.begin(), cumulative.end(), quantile);
            if (it == cumulative.end()) {
                --it;  // guards against quantile landing on/past the total mass due to floating-point rounding
            }
            return successors[static_cast<std::size_t>(it - cumulative.begin())];
        }
    };

    static SuccessorCdf buildSuccessorCdf(storm::storage::SparseMatrix<ValueType> const& matrix, uint64_t choice) {
        SuccessorCdf cdf;
        double sum = 0.0;
        for (auto const& entry : matrix.getRow(choice)) {
            if (!storm::utility::isZero(entry.getValue())) {
                sum += storm::utility::convertNumber<double>(entry.getValue());
                cdf.cumulative.push_back(sum);
                cdf.successors.push_back(entry.getColumn());
            }
        }
        return cdf;
    }

    /// Number of successors with nonzero real probability, without the overhead of building a full CDF.
    static uint64_t countNonzeroSuccessors(storm::storage::SparseMatrix<ValueType> const& matrix, uint64_t choice) {
        uint64_t count = 0;
        for (auto const& entry : matrix.getRow(choice)) {
            if (!storm::utility::isZero(entry.getValue())) {
                ++count;
            }
        }
        return count;
    }

    /*!
     * A dedicated RNG stream for one choice, deterministically derived from the run's seed and the choice
     * index via std::seed_seq (an algorithm fixed by the standard, so this is reproducible across
     * platforms/compilers). Giving each choice its own independent stream, instead of sharing one RNG
     * sequentially across choices, is what lets the sampleXxx() methods below sample choices in parallel:
     * a choice's result depends only on (seed, choice index), never on which thread computed it or how many
     * threads there were.
     */
    static std::mt19937_64 createChoiceRng(uint64_t seed, uint64_t choice) {
        std::seed_seq seedSeq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), static_cast<uint32_t>(choice),
                              static_cast<uint32_t>(choice >> 32)};
        return std::mt19937_64(seedSeq);
    }

    /*!
     * How parallelForEachChoice() will split `numChoices` choices: `numThreads` workers, each handling a
     * contiguous run of up to `choicesPerThread` choices (the last one may handle fewer). Exposed separately
     * so callers can report the actual thread count before sampling starts.
     */
    struct ThreadPlan {
        unsigned int numThreads;
        uint64_t choicesPerThread;
    };

    static ThreadPlan threadPlanFor(uint64_t numChoices) {
        unsigned int const maxThreads = std::max(1u, std::thread::hardware_concurrency());
        if (maxThreads <= 1 || numChoices <= 1) {
            return {1u, numChoices};
        }
        uint64_t const choicesPerThread = (numChoices + maxThreads - 1) / maxThreads;
        unsigned int const numThreads = static_cast<unsigned int>(std::min<uint64_t>(maxThreads, (numChoices + choicesPerThread - 1) / choicesPerThread));
        return {numThreads, choicesPerThread};
    }

    /*!
     * Runs `body(choice)` for every choice in [0, numChoices), split across threadPlanFor(numChoices)'s
     * worker threads by contiguous chunks (falling back to a plain sequential loop if that plan calls for a
     * single thread).
     * @note `body` must be safe to call concurrently for different choices: in practice this means it may
     * only touch shared state through per-choice-indexed writes (e.g. `result[choice] = ...`), and must
     * create its own RNG per choice (see createChoiceRng) rather than sharing one across choices.
     */
    template<typename Body>
    static void parallelForEachChoice(uint64_t numChoices, Body const& body) {
        ThreadPlan const plan = threadPlanFor(numChoices);
        if (plan.numThreads <= 1) {
            for (uint64_t choice = 0; choice < numChoices; ++choice) {
                body(choice);
            }
            return;
        }

        std::vector<std::thread> threads;
        threads.reserve(plan.numThreads);
        for (unsigned int t = 0; t < plan.numThreads; ++t) {
            uint64_t const begin = t * plan.choicesPerThread;
            uint64_t const end = std::min(numChoices, begin + plan.choicesPerThread);
            if (begin >= end) {
                break;
            }
            threads.emplace_back([begin, end, &body]() {
                for (uint64_t choice = begin; choice < end; ++choice) {
                    body(choice);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    /// Prints "Sampling N choices using M thread(s)." right before a parallel sampling pass starts.
    static void announceSampling(uint64_t numChoices) {
        unsigned int const numThreads = threadPlanFor(numChoices).numThreads;
        RUFFLE_LOG("Sampling " << numChoices << " choice" << (numChoices == 1 ? "" : "s") << " using " << numThreads << " thread"
                               << (numThreads == 1 ? "" : "s") << ".\n");
    }

    /// Per-choice sampling statistics gathered by the sampleXxx() methods below.
    struct ChoiceSamples {
        uint64_t totalCount{0};                                  ///< Total number of times the choice was sampled.
        std::unordered_map<uint64_t, uint64_t> successorCounts;  ///< Number of times each successor was sampled.

        void add(uint64_t sampledSuccessorState) {
            ++totalCount;
            if (auto emplaceRes = successorCounts.emplace(sampledSuccessorState, 1); !emplaceRes.second) {
                ++emplaceRes.first->second;
            }
        }
        uint64_t get(uint64_t successorState) const {
            if (auto const findRes = successorCounts.find(successorState); findRes != successorCounts.end()) {
                return findRes->second;
            }
            return 0;
        }
    };

    /*!
     * Draws exactly `numberOfSamples` samples per choice.
     * @param ensureFullCoverage If set, keeps sampling a choice past `numberOfSamples` until every one of
     * its successors (with real probability > 0) has been sampled at least once.
     */
    static std::vector<ChoiceSamples> sampleEachChoice(storm::storage::SparseMatrix<ValueType> const& matrix, uint64_t numberOfSamples, uint64_t const seed,
                                                       bool ensureFullCoverage) {
        uint64_t const numChoices = matrix.getRowCount();
        announceSampling(numChoices);
        std::vector<ChoiceSamples> result(numChoices);

        parallelForEachChoice(numChoices, [&](uint64_t currentChoice) {
            std::mt19937_64 rng = createChoiceRng(seed, currentChoice);
            std::uniform_real_distribution<double> uniformDistribution(0.0, 1.0);
            auto const cdf = buildSuccessorCdf(matrix, currentChoice);
            auto& currSamples = result[currentChoice];
            if (cdf.size() == 1) {
                currSamples.totalCount = numberOfSamples;
                currSamples.successorCounts.emplace(cdf.successor(0), numberOfSamples);
            } else {
                for (uint64_t i = 0; i < numberOfSamples; ++i) {
                    currSamples.add(cdf.sample(uniformDistribution(rng)));
                }
                while (ensureFullCoverage && currSamples.successorCounts.size() < cdf.size()) {
                    currSamples.add(cdf.sample(uniformDistribution(rng)));
                }
            }
        });
        return result;
    }

    /*!
     * Samples each choice (in growing batches) until the feasible L1 diameter of its Clopper-Pearson
     * confidence polytope drops to `delta` or below.
     * @param ensureFullCoverage If set, keeps sampling past that target until every one of the choice's
     * successors (with real probability > 0) has been sampled at least once.
     */
    static std::vector<ChoiceSamples> sampleMaxL1(storm::storage::SparseMatrix<ValueType> const& matrix, ValueType const lambda, ValueType const delta,
                                                  uint64_t const seed, bool ensureFullCoverage) {
        if (!(delta > zero)) {
            throw std::invalid_argument("Maximum L1 Clopper-Pearson interval width must be positive.");
        }

        uint64_t const numChoices = matrix.getRowCount();
        announceSampling(numChoices);
        std::vector<ChoiceSamples> result(numChoices);

        parallelForEachChoice(numChoices, [&](uint64_t currentChoice) {
            std::mt19937_64 rng = createChoiceRng(seed, currentChoice);
            std::uniform_real_distribution<double> uniformDistribution(0.0, 1.0);
            auto const cdf = buildSuccessorCdf(matrix, currentChoice);
            auto& currSamples = result[currentChoice];
            if (cdf.size() == 1) {
                currSamples.totalCount = 1;
                currSamples.successorCounts.emplace(cdf.successor(0), 1);
            } else {
                auto const componentFailureProbability = getComponentFailureProbability(lambda, cdf.size());

                for (uint64_t batchSize = 2048;; batchSize *= 2) {
                    for (uint64_t i = 0; i < batchSize; ++i) {
                        currSamples.add(cdf.sample(uniformDistribution(rng)));
                    }

                    ValueType lowerSum = zero;
                    ValueType upperSum = zero;
                    for (std::size_t i = 0; i < cdf.size(); ++i) {
                        auto const interval = getClopperPearsonInterval(currSamples.get(cdf.successor(i)), currSamples.totalCount, componentFailureProbability);
                        lowerSum += interval.lower();
                        upperSum += interval.upper();
                    }
                    if (auto const d = feasibleL1Diameter(lowerSum, upperSum);
                        d <= delta && (!ensureFullCoverage || currSamples.successorCounts.size() == cdf.size())) {
                        break;  // continue with next choice
                    } else if (batchSize >= 1u << 25) {
                        auto const log2Samples = std::bit_width(currSamples.totalCount);
                        RUFFLE_LOG("Choice #" << currentChoice << " with " << cdf.size() << " successors required approx. 2^" << log2Samples
                                              << " samples so far. Current L1=" << d << ".\n");
                    }
                }
            }
        });
        return result;
    }

    /*!
     * Samples each choice (in growing batches) until the L1 distance between the real distribution and the
     * empirical distribution over the drawn samples drops to `delta` or below. Unlike sampleMaxL1, this
     * compares directly against the known real distribution rather than a statistical confidence bound.
     * @param ensureFullCoverage If set, keeps sampling past that target until every one of the choice's
     * successors (with real probability > 0) has been sampled at least once.
     */
    static std::vector<ChoiceSamples> sampleUntilEmpiricalL1Distance(storm::storage::SparseMatrix<ValueType> const& matrix, double delta, uint64_t const seed,
                                                                     bool ensureFullCoverage) {
        if (!(delta > 0.0)) {
            throw std::invalid_argument("Maximum L1 distance must be positive.");
        }

        uint64_t const numChoices = matrix.getRowCount();
        announceSampling(numChoices);
        std::vector<ChoiceSamples> result(numChoices);

        parallelForEachChoice(numChoices, [&](uint64_t currentChoice) {
            std::mt19937_64 rng = createChoiceRng(seed, currentChoice);
            std::uniform_real_distribution<double> uniformDistribution(0.0, 1.0);
            auto const cdf = buildSuccessorCdf(matrix, currentChoice);
            auto& currSamples = result[currentChoice];
            if (cdf.size() == 1) {
                currSamples.totalCount = 1;
                currSamples.successorCounts.emplace(cdf.successor(0), 1);
            } else {
                for (uint64_t batchSize = 2048;; batchSize *= 2) {
                    for (uint64_t i = 0; i < batchSize; ++i) {
                        currSamples.add(cdf.sample(uniformDistribution(rng)));
                    }

                    double l1Distance = 0.0;
                    for (std::size_t i = 0; i < cdf.size(); ++i) {
                        double const empiricalProbability =
                            static_cast<double>(currSamples.get(cdf.successor(i))) / static_cast<double>(currSamples.totalCount);
                        l1Distance += std::abs(cdf.probability(i) - empiricalProbability);
                    }
                    if (l1Distance <= delta && (!ensureFullCoverage || currSamples.successorCounts.size() == cdf.size())) {
                        break;  // continue with next choice
                    }
                }
            }
        });
        return result;
    }

    /*!
     * Constructs a matrix builder for a new matrix over the same states/choices as `oldMatrix`, but with a
     * (possibly different) value type, ready to be filled via addNextValue in row-index order.
     */
    template<typename OutputValueType>
    static storm::storage::SparseMatrixBuilder<OutputValueType> createMatrixBuilder(storm::storage::SparseMatrix<ValueType> const& oldMatrix,
                                                                                    bool isDeterministicModel) {
        return storm::storage::SparseMatrixBuilder<OutputValueType>(oldMatrix.getRowCount(), oldMatrix.getColumnCount(), oldMatrix.getNonzeroEntryCount(), true,
                                                                    !isDeterministicModel, isDeterministicModel ? 0 : oldMatrix.getRowGroupCount());
    }

    /*!
     * Folds `value`'s bytes into `hash` using FNV-1a. Unlike std::hash (whose implementation is
     * unspecified), FNV-1a's algorithm is fully fixed, so equal hashes computed this way reliably mean
     * equal input even across different machines/compilers/standard library implementations.
     */
    static void fnv1aCombine(uint64_t& hash, uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (byte * 8)) & 0xffu;
            hash *= 0x100000001b3ULL;  // FNV-1a 64-bit prime
        }
    }

    static uint64_t doubleBits(double value) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /*!
     * @return A deterministic, cross-platform fingerprint (FNV-1a) of a matrix's transition probabilities
     * (both bounds, for an interval-valued matrix), computed in row-major order.
     */
    template<typename OutputValueType>
    static uint64_t hashTransitionProbabilities(storm::storage::SparseMatrix<OutputValueType> const& matrix) {
        uint64_t hash = 0xcbf29ce484222325ULL;  // FNV-1a 64-bit offset basis
        for (uint64_t row = 0; row < matrix.getRowCount(); ++row) {
            for (auto const& entry : matrix.getRow(row)) {
                fnv1aCombine(hash, row);
                fnv1aCombine(hash, entry.getColumn());
                if constexpr (storm::IsIntervalType<OutputValueType>) {
                    fnv1aCombine(hash, doubleBits(storm::utility::convertNumber<double>(entry.getValue().lower())));
                    fnv1aCombine(hash, doubleBits(storm::utility::convertNumber<double>(entry.getValue().upper())));
                } else {
                    fnv1aCombine(hash, doubleBits(storm::utility::convertNumber<double>(entry.getValue())));
                }
            }
        }
        return hash;
    }

    /*!
     * Assembles the final model around a freshly built transition matrix, carrying over state labeling,
     * choice labeling, state valuations and choice origins from the original model where present. Prints a
     * fingerprint hash of the output transition probabilities before handing the matrix off.
     */
    template<typename OutputValueType>
    static std::shared_ptr<storm::models::sparse::Model<OutputValueType>> finalizeModel(storm::models::sparse::Model<ValueType> const& model,
                                                                                        storm::storage::SparseMatrix<OutputValueType>&& matrix) {
        RUFFLE_LOG("Hash of output transition probabilities: 0x" << std::hex << std::setw(16) << std::setfill('0') << hashTransitionProbabilities(matrix)
                                                                 << std::dec << ".\n");

        storm::storage::sparse::ModelComponents<OutputValueType, storm::models::sparse::StandardRewardModel<OutputValueType>> components(
            std::move(matrix), model.getStateLabeling());
        // TODO: Carry over reward models; this needs a conversion from ValueType rewards to OutputValueType rewards.

        if (model.hasChoiceLabeling()) {
            components.choiceLabeling = storm::models::sparse::ChoiceLabeling(model.getChoiceLabeling());
        }
        if (model.hasStateValuations()) {
            components.stateValuations = model.getStateValuations();
        }
        if (model.hasChoiceOrigins()) {
            components.choiceOrigins = model.getChoiceOrigins();
        }

        return storm::utility::builder::buildModelFromComponents(model.getType(), std::move(components));
    }

    /// Builds the resulting interval model from per-choice sample statistics gathered by the caller.
    static std::shared_ptr<storm::models::sparse::Model<Interval>> buildLearnedIntervalModel(storm::models::sparse::Model<ValueType> const& model,
                                                                                             ValueType lambda, std::vector<ChoiceSamples> const& samples) {
        if (!(lambda > zero && lambda < one)) {
            throw std::invalid_argument("The local successor-distribution failure probability must be in the open interval (0, 1).");
        }

        auto const& oldMatrix = model.getTransitionMatrix();
        bool const isDeterministicModel = !model.isNondeterministicModel();
        auto builder = createMatrixBuilder<Interval>(oldMatrix, isDeterministicModel);
        auto const& stateChoiceIndices = oldMatrix.getRowGroupIndices();
        ValueType maximumFeasibleL1Diameter = zero;
        std::size_t numContainedStates = 0;

        for (std::size_t s = 0; s < model.getNumberOfStates(); ++s) {
            if (!isDeterministicModel) {
                builder.newRowGroup(stateChoiceIndices[s]);
            }

            // A state is "contained" if the true distribution of every one of its choices lies within the
            // learned intervals, i.e. for DTMCs this is a per-state check and for MDPs it requires all choices
            // of the state to be contained.
            bool stateContained = true;

            for (auto currentChoice = stateChoiceIndices[s]; currentChoice < stateChoiceIndices[s + 1]; ++currentChoice) {
                auto const row = oldMatrix.getRow(currentChoice);
                // Use the same (zero-filtered) notion of "single successor" as sampling did, so the [1, 1]
                // shortcut is only taken for choices that were actually sampled that way.
                bool const hasSingleSuccessor = countNonzeroSuccessors(oldMatrix, currentChoice) == 1;
                auto const componentFailureProbability = getComponentFailureProbability(lambda, row.getNumberOfEntries());
                auto const& choiceSamples = samples[currentChoice];

                ValueType lowerSum = zero;
                ValueType upperSum = zero;
                for (auto const& entry : row) {
                    Interval const estimatedInterval = estimateSuccessorInterval(hasSingleSuccessor, choiceSamples.get(entry.getColumn()),
                                                                                 choiceSamples.totalCount, componentFailureProbability);

                    if (entry.getValue() < estimatedInterval.lower() || entry.getValue() > estimatedInterval.upper()) {
                        stateContained = false;
                    }

                    lowerSum += estimatedInterval.lower();
                    upperSum += estimatedInterval.upper();
                    builder.addNextValue(currentChoice, entry.getColumn(), estimatedInterval);
                }
                maximumFeasibleL1Diameter = std::max(maximumFeasibleL1Diameter, feasibleL1Diameter(lowerSum, upperSum));
            }

            if (stateContained) {
                ++numContainedStates;
            }
        }

        RUFFLE_LOG("Maximum feasible L1-diameter over all learned choices: " << maximumFeasibleL1Diameter << ".\n");
        RUFFLE_LOG("Learned intervals contain the true distribution for "
                   << numContainedStates << " of " << model.getNumberOfStates() << " states ("
                   << (100.0 * static_cast<double>(numContainedStates) / static_cast<double>(model.getNumberOfStates())) << "%).\n");

        return finalizeModel(model, builder.build());
    }

    /*!
     * Builds an interval model by widening every concrete probability strictly between 0 and 1 into an
     * interval of the given width, centered at the original probability (clamped to stay within [0, 1]).
     * Deterministic: no sampling involved.
     * @param epsilon If set, every lower bound is additionally raised via applyEpsilonFloor.
     */
    static std::shared_ptr<storm::models::sparse::Model<Interval>> buildWidenedIntervalModel(storm::models::sparse::Model<ValueType> const& model,
                                                                                             ValueType const& delta, std::optional<ValueType> const& epsilon) {
        if (!(delta > zero)) {
            throw std::invalid_argument("Interval width must be positive.");
        }

        auto const& oldMatrix = model.getTransitionMatrix();
        bool const isDeterministicModel = !model.isNondeterministicModel();
        auto builder = createMatrixBuilder<Interval>(oldMatrix, isDeterministicModel);
        auto const& stateChoiceIndices = oldMatrix.getRowGroupIndices();

        for (std::size_t s = 0; s < model.getNumberOfStates(); ++s) {
            if (!isDeterministicModel) {
                builder.newRowGroup(stateChoiceIndices[s]);
            }
            for (auto currentChoice = stateChoiceIndices[s]; currentChoice < stateChoiceIndices[s + 1]; ++currentChoice) {
                for (auto const& entry : oldMatrix.getRow(currentChoice)) {
                    Interval const rawInterval = widenProbability(entry.getValue(), delta);
                    Interval const widenedInterval(applyEpsilonFloor(rawInterval.lower(), entry.getValue(), epsilon), rawInterval.upper());
                    builder.addNextValue(currentChoice, entry.getColumn(), widenedInterval);
                }
            }
        }

        return finalizeModel(model, builder.build());
    }

    /*!
     * Builds a model of the same value type as the input, replacing each choice's transition probabilities
     * by the empirical distribution observed over the given samples.
     */
    static std::shared_ptr<storm::models::sparse::Model<ValueType>> buildSampledDistributionModel(storm::models::sparse::Model<ValueType> const& model,
                                                                                                  std::vector<ChoiceSamples> const& samples) {
        auto const& oldMatrix = model.getTransitionMatrix();
        bool const isDeterministicModel = !model.isNondeterministicModel();
        auto builder = createMatrixBuilder<ValueType>(oldMatrix, isDeterministicModel);
        auto const& stateChoiceIndices = oldMatrix.getRowGroupIndices();

        for (std::size_t s = 0; s < model.getNumberOfStates(); ++s) {
            if (!isDeterministicModel) {
                builder.newRowGroup(stateChoiceIndices[s]);
            }
            for (auto currentChoice = stateChoiceIndices[s]; currentChoice < stateChoiceIndices[s + 1]; ++currentChoice) {
                auto const& choiceSamples = samples[currentChoice];
                for (auto const& entry : oldMatrix.getRow(currentChoice)) {
                    ValueType const empiricalProbability = storm::utility::convertNumber<ValueType>(choiceSamples.get(entry.getColumn())) /
                                                           storm::utility::convertNumber<ValueType>(choiceSamples.totalCount);
                    builder.addNextValue(currentChoice, entry.getColumn(), empiricalProbability);
                }
            }
        }

        return finalizeModel(model, builder.build());
    }
};

/*!
 * Learns an interval transition probability matrix by drawing exactly `numberOfSamples` samples per
 * state-action pair.
 * @param model The concrete input model (DTMC or MDP) to learn an IMDP from.
 * @param lambda Local successor-distribution failure probability: per state-action pair, the true
 * distribution lies within the reported successor intervals with probability at least `1 - lambda`.
 * @param numberOfSamples Number of samples drawn per state-action pair.
 * @param seed Seed for the random number generator.
 * @param ensureFullCoverage See LearningHelper::sampleEachChoice.
 * @return The learned interval model.
 */
template<typename ValueType>
std::shared_ptr<storm::models::sparse::Model<typename LearningHelper<ValueType>::Interval>> learnIMDPFromMDPByClopperPearsonUntilMaxSamples(
    storm::models::sparse::Model<ValueType> const& model, ValueType lambda, uint64_t numberOfSamples, uint64_t seed, bool ensureFullCoverage) {
    if (!model.isOfType(storm::models::ModelType::Dtmc) && !model.isOfType(storm::models::ModelType::Mdp)) {
        throw std::invalid_argument("We only support learning an interval model for DTMCs and MDPs as SUL.");
    }
    RUFFLE_LOG("Learning Clopper-Pearson intervals with local successor-distribution failure probability lambda="
               << lambda << " allocated over outgoing successors. Stop after " << numberOfSamples << " samples per choice.\n");

    return LearningHelper<ValueType>::buildLearnedIntervalModel(
        model, lambda, LearningHelper<ValueType>::sampleEachChoice(model.getTransitionMatrix(), numberOfSamples, seed, ensureFullCoverage));
}

/*!
 * Learns an IMDP by sampling each state-action pair (in growing batches) until the feasible L1 diameter of
 * its successor confidence polytope drops to `delta` or below.
 * @param model The concrete input model (DTMC or MDP) to learn an IMDP from.
 * @param lambda Local successor-distribution failure probability, see learnIMDPFromMDPByClopperPearsonUntilMaxSamples.
 * @param delta Target feasible L1 diameter.
 * @param seed Seed for the random number generator.
 * @param ensureFullCoverage See LearningHelper::sampleMaxL1.
 * @return The learned interval model.
 */
template<typename ValueType>
std::shared_ptr<storm::models::sparse::Model<typename LearningHelper<ValueType>::Interval>> learnIMDPFromMDPByClopperPearsonUntilL1Width(
    storm::models::sparse::Model<ValueType> const& model, ValueType lambda, ValueType delta, uint64_t seed, bool ensureFullCoverage) {
    if (!model.isOfType(storm::models::ModelType::Dtmc) && !model.isOfType(storm::models::ModelType::Mdp)) {
        throw std::invalid_argument("We only support learning an interval model for DTMCs and MDPs as SUL.");
    }
    RUFFLE_LOG("Learning Clopper-Pearson intervals with local successor-distribution failure probability lambda="
               << lambda << " allocated over outgoing successors. Stop when local L1 distance is below delta=" << delta << ".\n");

    return LearningHelper<ValueType>::buildLearnedIntervalModel(
        model, lambda, LearningHelper<ValueType>::sampleMaxL1(model.getTransitionMatrix(), lambda, delta, seed, ensureFullCoverage));
}

/*!
 * Replaces each concrete probability strictly between 0 and 1 by an interval of the given width, centered
 * at the original probability (clamped to [0, 1]). Deterministic: does not sample the model at all.
 * @param model The concrete input model to widen.
 * @param delta Width of the interval placed around each concrete probability.
 * @param epsilon See LearningHelper::applyEpsilonFloor.
 * @return The widened interval model.
 */
template<typename ValueType>
std::shared_ptr<storm::models::sparse::Model<typename LearningHelper<ValueType>::Interval>> widenModelIntervals(
    storm::models::sparse::Model<ValueType> const& model, ValueType delta, std::optional<ValueType> const& epsilon) {
    RUFFLE_LOG("Widening each concrete probability into an interval of width " << delta << ".\n");
    return LearningHelper<ValueType>::buildWidenedIntervalModel(model, delta, epsilon);
}

/*!
 * Replaces each choice's transition probabilities by the empirical distribution observed by drawing
 * exactly `numberOfSamples` samples per state-action pair from the real distribution.
 * @param model The concrete input model to resample.
 * @param numberOfSamples Number of samples drawn per state-action pair.
 * @param seed Seed for the random number generator.
 * @param ensureFullCoverage See LearningHelper::sampleEachChoice.
 * @return A perturbed point model of the same value type as `model`.
 */
template<typename ValueType>
std::shared_ptr<storm::models::sparse::Model<ValueType>> sampleModelDistributionUntilMaxSamples(storm::models::sparse::Model<ValueType> const& model,
                                                                                                uint64_t numberOfSamples, uint64_t seed,
                                                                                                bool ensureFullCoverage) {
    RUFFLE_LOG("Sampling " << numberOfSamples << " samples per state-action pair from the real distribution.\n");
    return LearningHelper<ValueType>::buildSampledDistributionModel(
        model, LearningHelper<ValueType>::sampleEachChoice(model.getTransitionMatrix(), numberOfSamples, seed, ensureFullCoverage));
}

/*!
 * Replaces each choice's transition probabilities by the empirical distribution observed by sampling (in
 * growing batches) until the L1 distance to the real distribution drops to `delta` or below.
 * @param model The concrete input model to resample.
 * @param delta Target L1 distance between the real and empirical distribution.
 * @param seed Seed for the random number generator.
 * @param ensureFullCoverage See LearningHelper::sampleUntilEmpiricalL1Distance.
 * @return A perturbed point model of the same value type as `model`.
 */
template<typename ValueType>
std::shared_ptr<storm::models::sparse::Model<ValueType>> sampleModelDistributionUntilL1Distance(storm::models::sparse::Model<ValueType> const& model,
                                                                                                double delta, uint64_t seed, bool ensureFullCoverage) {
    RUFFLE_LOG("Sampling from the real distribution per state-action pair until the L1 distance is at most delta=" << delta << ".\n");
    return LearningHelper<ValueType>::buildSampledDistributionModel(
        model, LearningHelper<ValueType>::sampleUntilEmpiricalL1Distance(model.getTransitionMatrix(), delta, seed, ensureFullCoverage));
}

}  // namespace ruffle
