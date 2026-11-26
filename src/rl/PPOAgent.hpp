#pragma once

#include "Adam.hpp"
#include "PolicyNetwork.hpp"
#include "RolloutBuffer.hpp"
#include "RunningStats.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <vector>

namespace gdai::rl {

enum class TrainingPreset {
    FastLearner,
    BalancedMaster,
    FineTunePolish
};

struct PPOConfig {
    std::size_t obsDim = 342;
    std::size_t obsNormDim = 0; // 0 = normalize all dims; >0 = normalize continuous prefix only
    std::size_t hidden1 = 128;
    std::size_t hidden2 = 64;
    std::size_t rolloutHorizon = 512;
    std::size_t minibatchSize = 64;
    std::size_t epochs = 4;

    float gamma = 0.99f;
    float gaeLambda = 0.95f;
    float clipEpsilon = 0.2f;
    float valueClipEpsilon = 0.2f;
    float valueLossCoef = 0.5f;
    float entropyCoefStart = 0.025f;
    float entropyCoefEnd = 0.003f;
    std::uint64_t entropyAnnealUpdates = 400;

    float learningRate = 3e-4f;
    float minLearningRate = 3e-5f;
    std::uint64_t lrAnnealUpdates = 1200;
    float maxGradNorm = 0.5f;
    float targetKL = 0.03f;

    // Self-Imitation Learning (SIL) from 2x new-best & completion runs
    float silCoef = 0.15f;
    std::size_t silBatches = 2;
    std::size_t maxEliteSamples = 1536;

    bool normalizeObs = true;
    std::uint32_t seed = 1337;

    static PPOConfig fromPreset(TrainingPreset preset, std::size_t obsDim = 342, std::size_t obsNormDim = 54);
};

struct ActionDecision {
    int action = 0;       // 0 = RELEASE, 1 = HOLD
    float holdProb = 0.5f;
    float logProb = 0.0f;
    float value = 0.0f;
    Vec normObs;
};

struct UpdateMetrics {
    float policyLoss = 0.0f;
    float valueLoss = 0.0f;
    float entropy = 0.0f;
    float approxKL = 0.0f;
    float clipFraction = 0.0f;
    float gradNorm = 0.0f;
    float explainedVariance = 0.0f;
    float learningRate = 0.0f;
    float entropyCoef = 0.0f;
    std::uint64_t updateCount = 0;
    bool klEarlyStopped = false;
};

struct EliteSample {
    Vec normObs;
    int action = 0;
    float discountedReturn = 0.0f;
    float episodeScore = 0.0f;
};

/// Self-contained C++23 Proximal Policy Optimization (PPO) + Self-Imitation Learning (SIL) agent.
class PPOAgent {
public:
    explicit PPOAgent(PPOConfig config = {});

    /// Select an action (stochastic when training=true, greedy argmax when training=false).
    ActionDecision selectAction(const Vec& rawObs, bool training);

    /// Read-only critic value estimate (used for cross-gamemode hot-swap bootstrapping).
    float estimateValue(const Vec& rawObs) const;

    /// Immediate mode: record a transition directly into the rollout buffer.
    std::optional<UpdateMetrics> recordTransition(
        const ActionDecision& decision,
        float reward,
        bool done,
        const Vec* nextRawObs = nullptr
    );

    /// Episode-deferred mode: stage a transition whose final reward depends on whether
    /// the episode dies (0x), sets a new best (2x), or completes.
    void stageTransition(
        const ActionDecision& decision,
        float rawReward,
        bool done,
        bool truncated,
        float bootstrapValue,
        std::uint64_t episodeId
    );

    /// Finalizes all staged transitions for episodeId with rewardMultiplier (0.0 on death without PB,
    /// 2.0 on new best, 1.0 on repeat completion), optionally stores elite trajectory for SIL,
    /// flushes into the rollout buffer, and triggers a PPO update if full.
    std::optional<UpdateMetrics> finalizeEpisode(
        std::uint64_t episodeId,
        float rewardMultiplier,
        float episodeScore,
        bool saveToElite,
        std::size_t fatalTrimSteps = 0,
        float bootstrapValue = 0.0f
    );

    /// Force an immediate PPO update on buffered transitions (if >= minibatchSize/2).
    std::optional<UpdateMetrics> forceUpdate(float bootstrapValue = 0.0f);

    /// Clear rollout & staged episode buffers without updating (e.g. on manual level exit).
    void clearBuffer();

    /// Reset network weights, optimizer state, elite buffer, and normalization stats.
    void reset(std::uint32_t newSeed = 0);

    /// Apply a hyperparameter preset without wiping trained weights.
    void applyPreset(TrainingPreset preset);

    void setLearningRate(float lr);

    bool saveModel(const std::filesystem::path& path) const;
    bool loadModel(const std::filesystem::path& path);

    const PPOConfig& config() const { return m_config; }
    PPOConfig& mutableConfig() { return m_config; }
    const PolicyNetwork& network() const { return m_net; }
    PolicyNetwork& mutableNetwork() { return m_net; }
    const RunningMeanStd& obsNorm() const { return m_obsNorm; }
    RunningMeanStd& mutableObsNorm() { return m_obsNorm; }
    const UpdateMetrics& lastMetrics() const { return m_lastMetrics; }
    std::uint64_t totalSteps() const { return m_totalSteps; }
    std::uint64_t updateCount() const { return m_updateCount; }
    void setStepsAndUpdates(std::uint64_t steps, std::uint64_t updates) {
        m_totalSteps = steps;
        m_updateCount = updates;
    }
    std::size_t bufferFill() const { return m_buffer.size() + m_pendingEpisode.size(); }
    std::size_t bufferCapacity() const { return m_buffer.capacity(); }
    std::size_t eliteCount() const { return m_eliteSamples.size(); }

    void setTrainingFrozen(bool frozen) { m_trainingFrozen = frozen; }
    bool isTrainingFrozen() const { return m_trainingFrozen; }

private:
    UpdateMetrics runPPOUpdate(float bootstrapValue);
    void runSelfImitationPass(float lr);
    void storeEliteTrajectory(const std::vector<Transition>& episode, float episodeScore, std::size_t fatalTrimSteps);
    float currentLearningRate() const;
    float currentEntropyCoef() const;

    PPOConfig m_config;
    PolicyNetwork m_net;
    AdamWOptimizer m_optim;
    RunningMeanStd m_obsNorm;
    RolloutBuffer m_buffer;
    std::vector<Transition> m_pendingEpisode;
    std::vector<EliteSample> m_eliteSamples;
    std::mt19937 m_rng;
    bool m_trainingFrozen = false;

    std::uint64_t m_totalSteps = 0;
    std::uint64_t m_updateCount = 0;
    UpdateMetrics m_lastMetrics;
};

} // namespace gdai::rl
