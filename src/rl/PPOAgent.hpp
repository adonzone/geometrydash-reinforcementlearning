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
    std::size_t obsDim = 240;
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

    bool normalizeObs = true;
    std::uint32_t seed = 1337;

    static PPOConfig fromPreset(TrainingPreset preset, std::size_t obsDim = 240);
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

/// Self-contained C++23 Proximal Policy Optimization (PPO) agent.
class PPOAgent {
public:
    explicit PPOAgent(PPOConfig config = {});

    /// Select an action (stochastic when training=true, greedy argmax when training=false).
    ActionDecision selectAction(const Vec& rawObs, bool training);

    /// Record a completed transition. Triggers a PPO update automatically once rolloutHorizon is reached.
    std::optional<UpdateMetrics> recordTransition(
        const ActionDecision& decision,
        float reward,
        bool done,
        const Vec* nextRawObs = nullptr
    );

    /// Force an immediate PPO update on buffered transitions (if >= minibatchSize).
    std::optional<UpdateMetrics> forceUpdate(float bootstrapValue = 0.0f);

    /// Clear rollout buffer without updating (e.g. on manual level exit).
    void clearBuffer();

    /// Reset network weights, optimizer state, and normalization stats.
    void reset(std::uint32_t newSeed = 0);

    /// Apply a hyperparameter preset without wiping trained weights.
    void applyPreset(TrainingPreset preset);

    void setLearningRate(float lr);

    bool saveModel(const std::filesystem::path& path) const;
    bool loadModel(const std::filesystem::path& path);

    const PPOConfig& config() const { return m_config; }
    PPOConfig& mutableConfig() { return m_config; }
    const PolicyNetwork& network() const { return m_net; }
    const UpdateMetrics& lastMetrics() const { return m_lastMetrics; }
    std::uint64_t totalSteps() const { return m_totalSteps; }
    std::uint64_t updateCount() const { return m_updateCount; }
    std::size_t bufferFill() const { return m_buffer.size(); }
    std::size_t bufferCapacity() const { return m_buffer.capacity(); }

private:
    UpdateMetrics runPPOUpdate(float bootstrapValue);
    float currentLearningRate() const;
    float currentEntropyCoef() const;

    PPOConfig m_config;
    PolicyNetwork m_net;
    AdamWOptimizer m_optim;
    RunningMeanStd m_obsNorm;
    RolloutBuffer m_buffer;
    std::mt19937 m_rng;

    std::uint64_t m_totalSteps = 0;
    std::uint64_t m_updateCount = 0;
    UpdateMetrics m_lastMetrics;
};

} // namespace gdai::rl
