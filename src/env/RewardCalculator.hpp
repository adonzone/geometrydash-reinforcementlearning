#pragma once

#include <algorithm>

namespace gdai::env {

struct RewardConfig {
    float forwardProgressScale = 0.035f; // per GD world unit moved forward (~1.05 per block)
    float frontierBonusScale = 0.05f;    // extra reward when breaking session max X
    float survivalStepBonus = 0.004f;    // small alive reward per decision tick
    float actionTogglePenalty = 0.002f;  // discourages useless button jitter
    float deathPenalty = -10.0f;         // terminal penalty on crash
    float newBestDeathRelief = 2.0f;     // softens death penalty if episode set a new PB
    float completionBonus = 50.0f;       // jackpot on level completion
};

/// Stateful per-episode reward shaper tuned for Geometry Dash.
class RewardCalculator {
public:
    explicit RewardCalculator(RewardConfig cfg = {}) : m_cfg(cfg) {}

    void onLevelStart() {
        m_sessionBestX = 0.0f;
        resetEpisode(0.0f);
    }

    void resetEpisode(float startX) {
        m_prevX = startX;
        m_prevAction = 0;
        m_brokeRecordThisEpisode = false;
    }

    float computeStepReward(float currentX, int action) {
        float reward = m_cfg.survivalStepBonus;
        const float dx = std::max(0.0f, currentX - m_prevX);
        reward += dx * m_cfg.forwardProgressScale;

        if (currentX > m_sessionBestX) {
            const float frontierDx = currentX - std::max(m_prevX, m_sessionBestX);
            reward += std::max(0.0f, frontierDx) * m_cfg.frontierBonusScale;
            m_sessionBestX = currentX;
            m_brokeRecordThisEpisode = true;
        }

        if (action != m_prevAction) {
            reward -= m_cfg.actionTogglePenalty;
        }

        m_prevX = currentX;
        m_prevAction = action;
        return reward;
    }

    float computeDeathReward() const {
        return m_cfg.deathPenalty + (m_brokeRecordThisEpisode ? m_cfg.newBestDeathRelief : 0.0f);
    }

    float computeCompletionReward() const {
        return m_cfg.completionBonus;
    }

    float sessionBestX() const { return m_sessionBestX; }
    RewardConfig& config() { return m_cfg; }
    const RewardConfig& config() const { return m_cfg; }

private:
    RewardConfig m_cfg;
    float m_prevX = 0.0f;
    float m_sessionBestX = 0.0f;
    int m_prevAction = 0;
    bool m_brokeRecordThisEpisode = false;
};

} // namespace gdai::env
