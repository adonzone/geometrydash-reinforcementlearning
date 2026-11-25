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
    : m_agent(rl::PPOConfig::fromPreset(rl::TrainingPreset::BalancedMaster, env::EncodedObservation::kTotalDim)) {}

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
    m_agent.setLearningRate(lrScaled * 1e-4f);

    refreshTimeScale();
}

std::filesystem::path AIManager::currentModelPath() const {
    const auto baseDir = Mod::get()->getSaveDir() / "models";
    if (m_perLevelModel && m_currentLevelId != 0) {
        return baseDir / fmt::format("level_{}.gdai", m_currentLevelId);
    }
    return baseDir / "master_general.gdai";
}

std::filesystem::path AIManager::statsCSVPath() const {
    return Mod::get()->getSaveDir() / fmt::format("stats_{}.csv", m_currentLevelId);
}

std::string AIManager::activeModelLabel() const {
    if (m_perLevelModel && m_currentLevelId != 0) {
        return fmt::format("Level #{}", m_currentLevelId);
    }
    return "Master Brain";
}

void AIManager::onEnterLevel(PlayLayer* playLayer) {
    m_inLevel = true;
    m_isPaused = false;
    m_episodeActive = true;
    m_currentlyHolding = false;
    m_holdDurationSec = 0.0f;
    m_tickCounter = 0;
    m_hasPendingDecision = false;
    m_hasObservation = false;

    if (playLayer && playLayer->m_level) {
        m_currentLevelId = playLayer->m_level->m_levelID.value();
        m_currentLevelName = playLayer->m_level->m_levelName;
    } else {
        m_currentLevelId = 0;
        m_currentLevelName = "Unknown Level";
    }

    m_stats.onLevelChanged(m_currentLevelId, m_currentLevelName);
    m_rewards.onLevelStart();
    m_agent.clearBuffer();

    if (m_perLevelModel) {
        loadCurrentModel();
    }

    refreshTimeScale();
    log::info("[GD-AI] Entered level '{}' (ID: {}) | Model: {}", m_currentLevelName, m_currentLevelId, activeModelLabel());
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

    // 1. Extract state & encode 240D egocentric observation
    const auto playerState = GameBridge::extractPlayerState(playLayer, m_currentlyHolding, m_holdDurationSec);
    const auto nearbyObjects = GameBridge::extractNearbyObjects(playLayer, playerState.x, playerState.y);
    const float groundY = playLayer->m_groundLayer ? playLayer->m_groundLayer->getPositionY() : 0.0f;

    m_latestObs = env::ObservationEncoder::encode(playerState, nearbyObjects, groundY);
    m_hasObservation = true;

    const float pct = GameBridge::currentPercent(playLayer);

    // 2. Close previous step transition with dense shaped reward
    if (m_hasPendingDecision && m_episodeActive) {
        const float stepReward = m_rewards.computeStepReward(playerState.x, m_pendingDecision.action);
        m_stats.onStep(
            pct,
            stepReward,
            m_pendingDecision.action,
            m_pendingDecision.holdProb,
            m_pendingDecision.value,
            playerState.mode
        );

        if (m_trainingMode) {
            auto maybeMetrics = m_agent.recordTransition(
                m_pendingDecision,
                stepReward,
                false,
                &m_latestObs.vec
            );
            if (maybeMetrics) {
                m_stats.onPPOUpdate(*maybeMetrics);
                maybeAutoSave(false);
            }
        }
    }

    // 3. Query PPO policy for next action and apply it to PlayerObject
    m_pendingDecision = m_agent.selectAction(m_latestObs.vec, m_trainingMode);
    m_hasPendingDecision = true;
    m_episodeActive = true;

    GameBridge::applyAction(playLayer, m_pendingDecision.action, m_currentlyHolding);
}

void AIManager::onPlayerDeath(PlayLayer* playLayer) {
    if (!m_inLevel || !m_episodeActive) return;

    GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
    m_holdDurationSec = 0.0f;
    m_episodeActive = false;

    const float pct = GameBridge::currentPercent(playLayer);
    const auto mode = m_stats.currentGamemode();

    if (m_aiEnabled && m_hasPendingDecision) {
        const float deathReward = m_rewards.computeDeathReward();
        m_stats.onStep(
            pct,
            deathReward,
            m_pendingDecision.action,
            m_pendingDecision.holdProb,
            m_pendingDecision.value,
            mode
        );
        if (m_trainingMode) {
            auto maybeMetrics = m_agent.recordTransition(m_pendingDecision, deathReward, true, nullptr);
            if (maybeMetrics) {
                m_stats.onPPOUpdate(*maybeMetrics);
            }
        }
    }

    m_hasPendingDecision = false;
    m_stats.onEpisodeEnd(pct, false, mode);
    maybeAutoSave(false);
}

void AIManager::onLevelComplete(PlayLayer* playLayer) {
    if (!m_inLevel || !m_episodeActive) return;

    GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
    m_holdDurationSec = 0.0f;
    m_episodeActive = false;

    const auto mode = m_stats.currentGamemode();
    if (m_aiEnabled && m_hasPendingDecision) {
        const float winReward = m_rewards.computeCompletionReward();
        m_stats.onStep(
            100.0f,
            winReward,
            m_pendingDecision.action,
            m_pendingDecision.holdProb,
            m_pendingDecision.value,
            mode
        );
        if (m_trainingMode) {
            auto maybeMetrics = m_agent.recordTransition(m_pendingDecision, winReward, true, nullptr);
            if (maybeMetrics) {
                m_stats.onPPOUpdate(*maybeMetrics);
            } else if (auto forced = m_agent.forceUpdate(0.0f)) {
                m_stats.onPPOUpdate(*forced);
            }
        }
    }

    m_hasPendingDecision = false;
    m_stats.onEpisodeEnd(100.0f, true, mode);
    maybeAutoSave(true);
    log::info("[GD-AI] LEVEL COMPLETED! Total clears: {}", m_stats.totalCompletions());
}

void AIManager::onResetLevel(PlayLayer* playLayer) {
    if (!m_inLevel) return;
    GameBridge::releaseAllInputs(playLayer, m_currentlyHolding);
    m_holdDurationSec = 0.0f;
    m_tickCounter = 0;
    m_hasPendingDecision = false;
    m_episodeActive = true;

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
    if (m_trainingMode && m_agent.updateCount() > m_lastSavedUpdate) {
        saveCurrentModel();
    }
    m_agent.clearBuffer();
    m_inLevel = false;
    m_isPaused = false;
    m_episodeActive = false;
    m_hasPendingDecision = false;
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
    if (m_trainingMode && m_agent.updateCount() > 0) {
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
    m_agent.applyPreset(preset);
}

bool AIManager::saveCurrentModel() {
    const auto path = currentModelPath();
    const bool ok = m_agent.saveModel(path);
    if (ok) {
        m_lastSavedBestPercent = m_stats.bestPercent();
        m_lastSavedUpdate = m_agent.updateCount();
        log::info("[GD-AI] Saved model checkpoint to {}", path.string());
    } else {
        log::error("[GD-AI] Failed to save model to {}", path.string());
    }
    return ok;
}

bool AIManager::loadCurrentModel() {
    const auto path = currentModelPath();
    if (!std::filesystem::exists(path)) {
        return false;
    }
    const bool ok = m_agent.loadModel(path);
    if (ok) {
        m_lastSavedUpdate = m_agent.updateCount();
        log::info("[GD-AI] Loaded model checkpoint from {} (updates: {}, steps: {})",
            path.string(), m_agent.updateCount(), m_agent.totalSteps());
    } else {
        log::warn("[GD-AI] Existing checkpoint at {} failed validation; keeping current weights", path.string());
    }
    return ok;
}

void AIManager::resetModelAndStats() {
    m_agent.reset();
    m_stats.resetAll();
    m_rewards.onLevelStart();
    m_lastSavedBestPercent = 0.0f;
    m_lastSavedUpdate = 0;
    log::info("[GD-AI] Reset neural network weights and training statistics.");
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
    const bool periodic = (m_agent.updateCount() >= m_lastSavedUpdate + 10);
    if (force || beatRecord || periodic) {
        saveCurrentModel();
    }
}

} // namespace gdai::game
