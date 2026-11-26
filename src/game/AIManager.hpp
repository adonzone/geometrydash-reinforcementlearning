#pragma once

#include "../env/Observation.hpp"
#include "../env/RewardCalculator.hpp"
#include "../env/StatsTracker.hpp"
#include "../rl/MultiModePPOAgent.hpp"

#include <Geode/Geode.hpp>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace gdai::game {

/// Central orchestrator linking Geometry Dash hooks, 8-gamemode hot-swapping PPO+SIL brains,
/// 0x-death / 2x-new-best reward shaping, persistence, and UI overlays.
class AIManager {
public:
    static AIManager& get();

    void initFromSettings();
    void syncFromSettings();

    // Lifecycle hooks called by PlayLayerHooks
    void onEnterLevel(PlayLayer* playLayer);
    void onPostUpdate(PlayLayer* playLayer, float dt);
    void onPlayerDeath(PlayLayer* playLayer);
    void onLevelComplete(PlayLayer* playLayer);
    void onResetLevel(PlayLayer* playLayer);
    void onPauseChanged(PlayLayer* playLayer, bool paused);
    void onLeaveLevel(PlayLayer* playLayer);

    // Controls exposed to UI & hotkeys
    bool isAIEnabled() const { return m_aiEnabled; }
    void setAIEnabled(bool enabled);

    bool isTrainingMode() const { return m_trainingMode; }
    void setTrainingMode(bool training);

    bool showHUD() const { return m_showHUD; }
    void setShowHUD(bool show);

    bool showVision() const { return m_showVision; }
    void setShowVision(bool show);

    bool perLevelModel() const { return m_perLevelModel; }
    void setPerLevelModel(bool perLevel);

    float trainingSpeed() const { return m_trainingSpeed; }
    void setTrainingSpeed(float speed);

    int decisionInterval() const { return m_decisionInterval; }
    void setDecisionInterval(int interval);

    // Model & preset management
    void applyPreset(rl::TrainingPreset preset);
    rl::TrainingPreset activePreset() const { return m_activePreset; }

    bool saveCurrentModel();
    bool loadCurrentModel();
    void resetModelAndStats();
    bool exportStatsCSV();

    void setModeTrainingFrozen(env::Gamemode mode, bool frozen) {
        m_multiAgent.setModeTrainingFrozen(mode, frozen);
    }
    bool isModeTrainingFrozen(env::Gamemode mode) const {
        return m_multiAgent.isModeTrainingFrozen(mode);
    }

    std::filesystem::path modelsDirectory() const;
    std::string activeModelPrefix() const;
    std::filesystem::path currentModelPath() const;
    std::filesystem::path masterBundlePath() const;
    std::filesystem::path statsCSVPath() const;
    std::string activeModelLabel() const;

    // Telemetry & vision accessors for UI
    const env::StatsTracker& stats() const { return m_stats; }
    const rl::MultiModePPOAgent& multiAgent() const { return m_multiAgent; }
    const rl::PPOAgent& agent() const { return m_multiAgent.activeAgent(); }
    env::Gamemode activeGamemode() const { return m_multiAgent.activeMode(); }
    bool brokeRecordThisEpisode() const { return m_rewards.brokeRecordThisEpisode(); }
    const env::EncodedObservation& latestObservation() const { return m_latestObs; }
    bool hasValidObservation() const { return m_hasObservation; }
    bool isInLevel() const { return m_inLevel; }

private:
    AIManager();
    void refreshTimeScale();
    void maybeAutoSave(bool force = false);

    rl::MultiModePPOAgent m_multiAgent;
    env::RewardCalculator m_rewards;
    env::StatsTracker m_stats;

    bool m_aiEnabled = true;
    bool m_trainingMode = true;
    bool m_showHUD = true;
    bool m_showVision = true;
    bool m_perLevelModel = false;
    bool m_autoSave = true;
    float m_trainingSpeed = 1.0f;
    int m_decisionInterval = 1;
    rl::TrainingPreset m_activePreset = rl::TrainingPreset::BalancedMaster;

    bool m_inLevel = false;
    bool m_isPaused = false;
    bool m_episodeActive = false;
    bool m_currentlyHolding = false;
    float m_holdDurationSec = 0.0f;
    float m_prevYVelocity = 0.0f;
    std::array<float, 4> m_recentActions{0.0f, 0.0f, 0.0f, 0.0f};
    int m_tickCounter = 0;

    bool m_hasTrackedHazard = false;
    float m_trackedHazardX = 0.0f;

    std::uint64_t m_episodeId = 1;
    int m_currentLevelId = 0;
    std::string m_currentLevelName = "None";

    bool m_hasPendingDecision = false;
    env::Gamemode m_pendingMode = env::Gamemode::Cube;
    rl::ActionDecision m_pendingDecision;

    bool m_hasObservation = false;
    env::EncodedObservation m_latestObs;

    float m_lastSavedBestPercent = 0.0f;
    std::uint64_t m_lastSavedUpdate = 0;
};

} // namespace gdai::game
