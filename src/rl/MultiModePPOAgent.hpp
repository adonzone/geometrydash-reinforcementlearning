#pragma once

#include "PPOAgent.hpp"
#include "../env/Observation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gdai::rl {

/// Manages 8 dedicated PPO+SIL models — one for each Geometry Dash gamemode
/// (Cube, Ship, Ball, UFO, Wave, Robot, Spider, Swing) — with instant O(1) hot-swapping
/// on gamemode portals and coordinated multi-mode episode finalization.
class MultiModePPOAgent {
public:
    static constexpr std::size_t kModeCount = static_cast<std::size_t>(env::Gamemode::Count);

    explicit MultiModePPOAgent(
        TrainingPreset preset = TrainingPreset::BalancedMaster,
        std::size_t obsDim = env::EncodedObservation::kTotalDim,
        std::size_t obsNormDim = env::EncodedObservation::kContinuousNormDim
    );

    /// Instantly hot-swaps the active model to the requested gamemode (O(1), zero I/O).
    /// Returns true if a mode switch actually occurred.
    bool setActiveMode(env::Gamemode mode);
    env::Gamemode activeMode() const { return m_activeMode; }

    PPOAgent& agentFor(env::Gamemode mode);
    const PPOAgent& agentFor(env::Gamemode mode) const;
    PPOAgent& activeAgent() { return agentFor(m_activeMode); }
    const PPOAgent& activeAgent() const { return agentFor(m_activeMode); }

    /// Hot-swaps to `mode` if needed and queries that gamemode's dedicated neural network.
    ActionDecision selectAction(env::Gamemode mode, const Vec& rawObs, bool training);

    /// Read-only critic value estimate from a specific gamemode's model (for portal bootstrapping).
    float estimateValue(env::Gamemode mode, const Vec& rawObs) const;

    /// Stages a transition into the exact gamemode model that produced `decision`.
    void stageTransition(
        env::Gamemode mode,
        const ActionDecision& decision,
        float rawReward,
        bool done,
        bool truncated,
        float bootstrapValue,
        std::uint64_t episodeId
    );

    /// Finalizes `episodeId` across every gamemode model that participated in that episode.
    /// Applies the episode-level reward multiplier (0x on death without PB, 2x on new best),
    /// updates elite SIL buffers, and returns any PPO update metrics triggered.
    std::vector<UpdateMetrics> finalizeEpisode(
        std::uint64_t episodeId,
        float rewardMultiplier,
        float episodeScore,
        bool saveToElite,
        std::size_t fatalTrimSteps = 0
    );

    /// Forces an update on the active gamemode's model (e.g., on level completion).
    std::optional<UpdateMetrics> forceUpdateActive(float bootstrapValue = 0.0f);

    void clearAllBuffers();
    void resetAll(std::uint32_t baseSeed = 1337);
    void applyPresetAll(TrainingPreset preset);
    void setLearningRateAll(float lr);
    void setModeTrainingFrozen(env::Gamemode mode, bool frozen);
    bool isModeTrainingFrozen(env::Gamemode mode) const;

    /// Master bundle checkpoint path: `<dir>/<prefix>_master.gdai`
    static std::filesystem::path masterBundlePathFor(
        const std::filesystem::path& dir,
        const std::string& prefix
    );

    /// Saves a single unified master bundle containing all 8 gamemode brains.
    bool saveMasterBundle(const std::filesystem::path& bundlePath) const;

    /// Loads a master bundle containing all 8 gamemode brains.
    bool loadMasterBundle(const std::filesystem::path& bundlePath);

    /// Per-gamemode checkpoint path: `<dir>/<prefix>_<modeSlug>.gdai`
    static std::filesystem::path modelPathFor(
        const std::filesystem::path& dir,
        const std::string& prefix,
        env::Gamemode mode
    );

    /// Saves all gamemode models that have been trained (or all if saveUntrained=true).
    /// Returns the number of gamemode models saved.
    std::size_t saveAllModels(
        const std::filesystem::path& dir,
        const std::string& prefix,
        bool saveUntrained = false
    ) const;

    /// Loads all existing per-gamemode models into memory so gameplay hot-swaps are instant.
    /// Returns the number of gamemode models successfully loaded.
    std::size_t loadAllModels(
        const std::filesystem::path& dir,
        const std::string& prefix
    );

    std::uint64_t hotSwapCount() const { return m_hotSwapCount; }
    void resetHotSwapCount() { m_hotSwapCount = 0; }

    std::uint64_t totalStepsAll() const;
    std::uint64_t totalUpdatesAll() const;
    std::size_t totalParametersAll() const;
    std::size_t trainedModeCount() const;

private:
    static std::size_t toIndex(env::Gamemode mode);
    static PPOConfig makeModeConfig(
        env::Gamemode mode,
        TrainingPreset preset,
        std::size_t obsDim,
        std::size_t obsNormDim
    );

    std::vector<PPOAgent> m_agents;
    env::Gamemode m_activeMode = env::Gamemode::Cube;
    std::uint64_t m_hotSwapCount = 0;
};

} // namespace gdai::rl
