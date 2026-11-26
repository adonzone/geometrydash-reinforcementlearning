#include "AIManager.hpp"
#include "GameBridge.hpp"

#include <algorithm>
#include <cmath>

using namespace geode::prelude;

namespace gdai::game {

AIManager& AIManager::get() {
    static AIManager instance;
    return instance;
}

AIManager::AIManager()
    : m_multiAgent(
          rl::TrainingPreset::BalancedMaster,
          env::EncodedObservation::kTotalDim,
          env::EncodedObservation::kContinuousNormDim
      ) {}

void AIManager::initFromSettings() {
    syncFromSettings();
    loadCurrentModel();
}

void AIManager::syncFromSettings() {
    auto* mod = Mod::get();
    if (!mod) return;

    m_aiEnabled = mod->getSettingValue<bool>("ai-enabled");
    m_trainingMode = mod->getSettingValue<bool>("training-mode");
    m_showHUD = mod->getSettingValue<bool>("show-hud");
    m_showVision = mod->getSettingValue<bool>("show-vision");
    m_perLevelModel = mod->getSettingValue<bool>("per-level-model");
    m_autoSave = mod->getSettingValue<bool>("auto-save");
    m_trainingSpeed = static_cast<float>(mod->getSettingValue<double>("training-speed"));
    m_decisionInterval = static_cast<int>(mod->getSettingValue<int64_t>("decision-interval"));

    const float lrScaled = static_cast<float>(mod->getSettingValue<double>("learning-rate"));
    m_multiAgent.setLearningRateAll(lrScaled * 1e-4f);

    refreshTimeScale();
}

std::filesystem::path AIManager::modelsDirectory() const {
    return std::filesystem::path(Mod::get()->getSaveDir().native()) / "models";
}

std::string AIManager::activeModelPrefix() const {
    if (m_perLevelModel && m_currentLevelId != 0) {
        return fmt::format("level_{}", m_currentLevelId);
    }
    return "master";
}

std::filesystem::path AIManager::currentModelPath() const {
    return rl::MultiModePPOAgent::modelPathFor(
        modelsDirectory(),
        activeModelPrefix(),
        m_multiAgent.activeMode()
    );
}

std::filesystem::path AIManager::masterBundlePath() const {
    return rl::MultiModePPOAgent::masterBundlePathFor(
        modelsDirectory(),
        activeModelPrefix()
    );
}

std::filesystem::path AIManager::statsCSVPath() const {
    return std::filesystem::path(Mod::get()->getSaveDir().native()) / fmt::format("stats_{}.csv", m_currentLevelId);
}

std::string AIManager::activeModelLabel() const {
    const char* modeStr = env::gamemodeName(m_multiAgent.activeMode());
    if (m_perLevelModel && m_currentLevelId != 0) {
        return fmt::format("Lvl #{} [{}]", m_currentLevelId, modeStr);
    }
    return fmt::format("Master [{}]", modeStr);
}

void AIManager::onEnterLevel(PlayLayer* playLayer) {
    m_inLevel = true;
    m_isPaused = false;
    m_episodeActive = true;
    m_currentlyHolding = false;
    m_holdDurationSec = 0.0f;
    m_prevYVelocity = 0.0f;
    m_recentActions = {0.0f, 0.0f, 0.0f, 0.0f};
    m_tickCounter = 0;
    m_hasTrackedHazard = false;
    m_trackedHazardX = 0.0f;
    m_hasPendingDecision = false;
    m_hasObservation = false;
    ++m_episodeId;

    if (playLayer && playLayer->m_level) {
        m_currentLevelId = playLayer->m_level->m_levelID.value();
        m_currentLevelName = playLayer->m_level->m_levelName;
    } else {
        m_currentLevelId = 0;
        m_currentLevelName = "Unknown Level";
    }

    m_stats.onLevelChanged(m_currentLevelId, m_currentLevelName);
    m_rewards.onLevelStart();
    m_multiAgent.clearAllBuffers();
    m_multiAgent.resetHotSwapCount();

    if (m_perLevelModel) {
        loadCurrentModel();
    }

    refreshTimeScale();
    log::info(
        "[GD-AI] Entered level '{}' (ID: {}) | Hot-Swap Brain: {} (8 gamemode models ready)",
        m_currentLevelName,
        m_currentLevelId,
        activeModelLabel()
    );
}

void AIManager::onPostUpdate(PlayLayer* playLayer, float dt) {
    if (!m_inLevel || m_isPaused || !playLayer || !playLayer->m_player1) return;
    if (playLayer->m_player1->m_isDead || playLayer->m_hasCompletedLevel) return;

    if (!m_aiEnabled) {
        GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
        return;
    }

    if (m_currentlyHolding) {
        m_holdDurationSec += dt;
    } else {
        m_holdDurationSec = 0.0f;
    }

    ++m_tickCounter;
    if (m_tickCounter < std::max(1, m_decisionInterval)) {
        return;
    }
    m_tickCounter = 0;

    // 1. Extract enriched player state & encode 342D egocentric observation
    auto playerState = GameBridge::extractPlayerState(playLayer, m_currentlyHolding, m_holdDurationSec);
    playerState.prevYVelocity = m_prevYVelocity;
    playerState.recentActions = m_recentActions;
    m_prevYVelocity = playerState.yVelocity;

    const auto nearbyObjects = GameBridge::extractNearbyObjects(playLayer, playerState.x, playerState.y);
    const float groundY = playLayer->m_groundLayer ? playLayer->m_groundLayer->getPositionY() : 0.0f;

    m_latestObs = env::ObservationEncoder::encode(playerState, nearbyObjects, groundY);
    m_hasObservation = true;

    // Detect clean hazard clearance (player safely passed the previously tracked nearest hazard)
    bool clearedHazard = false;
    if (m_hasTrackedHazard && playerState.x > m_trackedHazardX + 12.0f) {
        clearedHazard = true;
        m_hasTrackedHazard = false;
    }
    if (m_latestObs.targets.hasHazard1 && m_latestObs.targets.hazard1Dx > 0.0f && m_latestObs.targets.hazard1Dx < 180.0f) {
        const float nextHazX = playerState.x + m_latestObs.targets.hazard1Dx;
        if (!m_hasTrackedHazard || nextHazX < m_trackedHazardX) {
            m_trackedHazardX = nextHazX;
            m_hasTrackedHazard = true;
        }
    }

    const float pct = GameBridge::currentPercent(playLayer);

    // 2. Stage previous transition on the gamemode model that took it (handling hot-swaps cleanly!)
    if (m_hasPendingDecision && m_episodeActive) {
        const float rawStepReward = m_rewards.computeStepReward(
            playerState.x,
            m_pendingDecision.action,
            clearedHazard
        );
        m_stats.onStep(
            pct,
            rawStepReward,
            m_pendingDecision.action,
            m_pendingDecision.holdProb,
            m_pendingDecision.value,
            playerState.mode
        );

        if (m_trainingMode) {
            const bool modeSwapped = (playerState.mode != m_pendingMode);
            const float portalBootstrap = modeSwapped
                ? m_multiAgent.estimateValue(playerState.mode, m_latestObs.vec)
                : 0.0f;

            m_multiAgent.stageTransition(
                m_pendingMode,
                m_pendingDecision,
                rawStepReward,
                false,
                modeSwapped,
                portalBootstrap,
                m_episodeId
            );
        }
    }

    // 3. Hot-swap to current gamemode's dedicated model and select next action
    if (m_multiAgent.setActiveMode(playerState.mode)) {
        log::debug(
            "[GD-AI] Hot-swapped brain -> {} model (swap #{})",
            env::gamemodeName(playerState.mode),
            m_multiAgent.hotSwapCount()
        );
    }

    m_pendingDecision = m_multiAgent.selectAction(playerState.mode, m_latestObs.vec, m_trainingMode);
    m_pendingMode = playerState.mode;
    m_hasPendingDecision = true;
    m_episodeActive = true;

    // Shift 4-step action history buffer (0.0=Release, 0.5=Click, 1.0=Hold)
    m_recentActions[3] = m_recentActions[2];
    m_recentActions[2] = m_recentActions[1];
    m_recentActions[1] = m_recentActions[0];
    m_recentActions[0] = (m_pendingDecision.action == 2) ? 1.0f : ((m_pendingDecision.action == 1) ? 0.5f : 0.0f);

    GameBridge::applyAction(playLayer, m_pendingDecision.action, m_currentlyHolding);
}

void AIManager::onPlayerDeath(PlayLayer* playLayer) {
    if (!m_inLevel || !m_episodeActive) return;

    GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
    m_holdDurationSec = 0.0f;
    m_episodeActive = false;

    const float pct = GameBridge::currentPercent(playLayer);
    const auto mode = m_hasPendingDecision ? m_pendingMode : m_stats.currentGamemode();
    const auto resolution = m_rewards.resolveDeath();

    if (m_aiEnabled && m_hasPendingDecision) {
        m_stats.onStep(
            pct,
            resolution.terminalReward,
            m_pendingDecision.action,
            m_pendingDecision.holdProb,
            m_pendingDecision.value,
            mode
        );

        if (m_trainingMode) {
            // Stage terminal death transition with 0.0f reward (no reward on death)
            m_multiAgent.stageTransition(
                m_pendingMode,
                m_pendingDecision,
                resolution.terminalReward,
                true,
                false,
                0.0f,
                m_episodeId
            );

            // Finalize episode across all gamemodes that participated:
            // - 0x (no reward) if it died without a new best
            // - 2x reward + elite SIL trajectory if it achieved a new best!
            const auto updates = m_multiAgent.finalizeEpisode(
                m_episodeId,
                resolution.rewardMultiplier,
                pct,
                resolution.saveToElite,
                resolution.fatalTrimSteps
            );
            for (const auto& u : updates) {
                m_stats.onPPOUpdate(u);
            }
        }
    }

    m_hasPendingDecision = false;
    m_hasTrackedHazard = false;
    m_stats.onEpisodeEnd(pct, false, mode, resolution.finalEpisodeReward);
    maybeAutoSave(resolution.achievedNewBest);
}

void AIManager::onLevelComplete(PlayLayer* playLayer) {
    if (!m_inLevel || !m_episodeActive) return;

    GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
    m_holdDurationSec = 0.0f;
    m_episodeActive = false;

    const auto mode = m_hasPendingDecision ? m_pendingMode : m_stats.currentGamemode();
    const auto resolution = m_rewards.resolveCompletion();

    if (m_aiEnabled && m_hasPendingDecision) {
        m_stats.onStep(
            100.0f,
            resolution.terminalReward,
            m_pendingDecision.action,
            m_pendingDecision.holdProb,
            m_pendingDecision.value,
            mode
        );

        if (m_trainingMode) {
            m_multiAgent.stageTransition(
                m_pendingMode,
                m_pendingDecision,
                resolution.terminalReward,
                true,
                false,
                0.0f,
                m_episodeId
            );

            const auto updates = m_multiAgent.finalizeEpisode(
                m_episodeId,
                resolution.rewardMultiplier,
                100.0f,
                resolution.saveToElite,
                0
            );
            if (!updates.empty()) {
                for (const auto& u : updates) {
                    m_stats.onPPOUpdate(u);
                }
            } else if (auto forced = m_multiAgent.forceUpdateActive(0.0f)) {
                m_stats.onPPOUpdate(*forced);
            }
        }
    }

    m_hasPendingDecision = false;
    m_hasTrackedHazard = false;
    m_stats.onEpisodeEnd(100.0f, true, mode, resolution.finalEpisodeReward);
    maybeAutoSave(true);
    log::info(
        "[GD-AI] LEVEL COMPLETED! Reward: {:.1f} ({}x) | Total clears: {}",
        resolution.finalEpisodeReward,
        resolution.rewardMultiplier,
        m_stats.totalCompletions()
    );
}

void AIManager::onResetLevel(PlayLayer* playLayer) {
    if (!m_inLevel) return;
    GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
    m_holdDurationSec = 0.0f;
    m_prevYVelocity = 0.0f;
    m_recentActions = {0.0f, 0.0f, 0.0f, 0.0f};
    m_tickCounter = 0;
    m_hasTrackedHazard = false;
    m_trackedHazardX = 0.0f;
    m_hasPendingDecision = false;
    m_episodeActive = true;
    ++m_episodeId;

    const float startX = (playLayer && playLayer->m_player1) ? playLayer->m_player1->getPositionX() : 0.0f;
    m_rewards.resetEpisode(startX);
    refreshTimeScale();
}

void AIManager::onPauseChanged(PlayLayer* playLayer, bool paused) {
    m_isPaused = paused;
    if (paused) {
        GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
        GameBridge::setGameTimeScale(1.0f);
    } else {
        refreshTimeScale();
    }
}

void AIManager::onLeaveLevel(PlayLayer* playLayer) {
    GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
    GameBridge::setGameTimeScale(1.0f);
    if (m_trainingMode && m_multiAgent.totalUpdatesAll() > m_lastSavedUpdate) {
        saveCurrentModel();
    }
    m_multiAgent.clearAllBuffers();
    m_inLevel = false;
    m_isPaused = false;
    m_episodeActive = false;
    m_hasPendingDecision = false;
    m_hasTrackedHazard = false;
    m_hasObservation = false;
}

void AIManager::setAIEnabled(bool enabled) {
    m_aiEnabled = enabled;
    if (auto* mod = Mod::get()) mod->setSettingValue("ai-enabled", enabled);
    if (!enabled) {
        if (auto* pl = PlayLayer::get()) {
            GameBridge::releaseAllInputs(pl, m_currentlyHolding);
        }
    }
    refreshTimeScale();
}

void AIManager::setTrainingMode(bool training) {
    m_trainingMode = training;
    if (auto* mod = Mod::get()) mod->setSettingValue("training-mode", training);
    refreshTimeScale();
}

void AIManager::setShowHUD(bool show) {
    m_showHUD = show;
    if (auto* mod = Mod::get()) mod->setSettingValue("show-hud", show);
}

void AIManager::setShowVision(bool show) {
    m_showVision = show;
    if (auto* mod = Mod::get()) mod->setSettingValue("show-vision", show);
}

void AIManager::setPerLevelModel(bool perLevel) {
    if (m_perLevelModel == perLevel) return;
    if (m_trainingMode && m_multiAgent.totalUpdatesAll() > 0) {
        saveCurrentModel();
    }
    m_perLevelModel = perLevel;
    if (auto* mod = Mod::get()) mod->setSettingValue("per-level-model", perLevel);
    loadCurrentModel();
}

void AIManager::setTrainingSpeed(float speed) {
    m_trainingSpeed = std::clamp(speed, 0.5f, 4.0f);
    if (auto* mod = Mod::get()) mod->setSettingValue<double>("training-speed", static_cast<double>(m_trainingSpeed));
    refreshTimeScale();
}

void AIManager::setDecisionInterval(int interval) {
    m_decisionInterval = std::clamp(interval, 1, 6);
    if (auto* mod = Mod::get()) mod->setSettingValue<int64_t>("decision-interval", m_decisionInterval);
}

void AIManager::applyPreset(rl::TrainingPreset preset) {
    m_activePreset = preset;
    m_multiAgent.applyPresetAll(preset);
}

bool AIManager::saveCurrentModel() {
    const auto dir = modelsDirectory();
    const auto prefix = activeModelPrefix();
    std::size_t saved = m_multiAgent.saveAllModels(dir, prefix, true);
    const bool masterSaved = m_multiAgent.saveMasterBundle(masterBundlePath());
    if (saved > 0 || masterSaved) {
        m_lastSavedBestPercent = m_stats.bestPercent();
        m_lastSavedUpdate = m_multiAgent.totalUpdatesAll();
        log::info(
            "[GD-AI] Saved {} gamemode checkpoints & master bundle under {}/{}",
            saved,
            dir.string(),
            prefix
        );
        return true;
    }
    log::error("[GD-AI] Failed to save gamemode models to {}", dir.string());
    return false;
}

bool AIManager::loadCurrentModel() {
    const auto dir = modelsDirectory();
    const auto prefix = activeModelPrefix();
    const auto masterPath = masterBundlePath();

    if (std::filesystem::exists(masterPath) && m_multiAgent.loadMasterBundle(masterPath)) {
        m_lastSavedUpdate = m_multiAgent.totalUpdatesAll();
        log::info(
            "[GD-AI] Loaded master multi-mode bundle from {} (total updates: {}, steps: {})",
            masterPath.string(),
            m_multiAgent.totalUpdatesAll(),
            m_multiAgent.totalStepsAll()
        );
        return true;
    }

    const std::size_t loaded = m_multiAgent.loadAllModels(dir, prefix);
    if (loaded > 0) {
        m_lastSavedUpdate = m_multiAgent.totalUpdatesAll();
        log::info(
            "[GD-AI] Loaded {} gamemode model(s) from {}/{}_*.gdai (total updates: {}, steps: {})",
            loaded,
            dir.string(),
            prefix,
            m_multiAgent.totalUpdatesAll(),
            m_multiAgent.totalStepsAll()
        );
        return true;
    }
    return false;
}

void AIManager::resetModelAndStats() {
    m_multiAgent.resetAll();
    m_stats.resetAll();
    m_rewards.onLevelStart();
    m_lastSavedBestPercent = 0.0f;
    m_lastSavedUpdate = 0;
    log::info("[GD-AI] Reset all 8 gamemode neural networks and training statistics.");
}

bool AIManager::exportStatsCSV() {
    const auto path = statsCSVPath();
    const bool ok = m_stats.exportCSV(path);
    if (ok) {
        log::info("[GD-AI] Exported training statistics to {}", path.string());
    }
    return ok;
}

void AIManager::refreshTimeScale() {
    if (m_inLevel && !m_isPaused && m_aiEnabled && m_trainingMode) {
        GameBridge::setGameTimeScale(m_trainingSpeed);
    } else {
        GameBridge::setGameTimeScale(1.0f);
    }
}

void AIManager::maybeAutoSave(bool force) {
    if (!m_autoSave) return;
    const bool beatRecord = m_stats.bestPercent() > m_lastSavedBestPercent + 2.0f;
    const bool periodic = (m_multiAgent.totalUpdatesAll() >= m_lastSavedUpdate + 10);
    if (force || beatRecord || periodic) {
        saveCurrentModel();
    }
}

} // namespace gdai::game
