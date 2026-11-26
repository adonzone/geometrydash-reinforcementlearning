#pragma once

#include <algorithm>
#include <cstddef>

namespace gdai::env {

struct RewardConfig {
    float forwardProgressScale = 0.04f;  // per GD world unit moved forward (~1.2 per block)
    float survivalStepBonus = 0.005f;    // small alive reward per decision tick
    float hazardClearBonus = 0.35f;      // bonus when cleanly passing a hazard ahead
    float actionTogglePenalty = 0.0015f; // discourages useless button jitter
    float newBestMultiplier = 2.0f;      // 2x episode reward multiplier when beating best distance
    float deathReward = -1.0f;           // terminal collision penalty (-1.0) so critic learns crash action
    float completionBonus = 50.0f;       // bonus on 100% level completion
    float minNewBestDeltaX = 5.0f;       // minimum X advance past previous best to count as a new best
    std::size_t rollingPBStagnationEpisodes = 30; // episodes without PB before soft-floor recovery activates
    float rollingRecoveryFactor = 0.90f;          // soft-floor multiplier to prevent policy starvation
};

struct EpisodeRewardResolution {
    float rewardMultiplier = 0.0f;  // 0.0 on death without PB, 2.0 on new best, 1.0 on repeat completion
    float terminalReward = -1.0f;   // -1.0 on death, completionBonus on completion
    float finalEpisodeReward = 0.0f;
    bool achievedNewBest = false;
    bool saveToElite = false;
    std::size_t fatalTrimSteps = 0; // trim last few crash frames from elite trajectory when dying after a PB
};

/// Stateful per-episode reward shaper implementing:
/// - No reward if the agent dies (0x trajectory reward, but -1.0 terminal collision penalty on the crash tick).
/// - 2x reward multiplier whenever the agent achieves a new personal best distance.
/// - Session Personal Best with Rolling Recovery (soft floor prevents starvation after long plateaus).
/// - Full progress + completion bonus on 100% level clear (2x if it is also a new best).
class RewardCalculator {
public:
    explicit RewardCalculator(RewardConfig cfg = {}) : m_cfg(cfg) {}

    void onLevelStart() {
        m_sessionBestX = 0.0f;
        m_stagnantEpisodes = 0;
        resetEpisode(0.0f);
    }

    void resetEpisode(float startX) {
        m_prevX = startX;
        m_episodeStartBestX = m_sessionBestX;
        m_prevAction = 0;
        m_brokeRecordThisEpisode = false;
        m_episodeRawReward = 0.0f;
    }

    /// Computes provisional alive-step reward (finalized on death or completion).
    float computeStepReward(float currentX, int action, bool clearedHazard = false) {
        float reward = m_cfg.survivalStepBonus;
        const float dx = std::max(0.0f, currentX - m_prevX);
        reward += dx * m_cfg.forwardProgressScale;

        if (clearedHazard) {
            reward += m_cfg.hazardClearBonus;
        }

        if (action != m_prevAction) {
            reward -= m_cfg.actionTogglePenalty;
        }
        reward = std::max(0.0f, reward);

        if (currentX > m_sessionBestX) {
            if (currentX >= m_episodeStartBestX + m_cfg.minNewBestDeltaX) {
                m_brokeRecordThisEpisode = true;
            }
            m_sessionBestX = currentX;
        }

        m_prevX = currentX;
        m_prevAction = action;
        m_episodeRawReward += reward;
        return reward;
    }

    /// Resolves episode reward on player death:
    /// - No reward (0x trajectory) if it dies without setting a new best, plus -1.0 terminal crash penalty.
    /// - 2x reward if it set a new best before dying.
    /// - Applies rolling recovery if stagnant for multiple episodes.
    EpisodeRewardResolution resolveDeath() {
        EpisodeRewardResolution res;
        res.achievedNewBest = m_brokeRecordThisEpisode;
        res.terminalReward = m_cfg.deathReward; // -1.0 terminal penalty on collision frame
        if (m_brokeRecordThisEpisode) {
            m_stagnantEpisodes = 0;
            res.rewardMultiplier = m_cfg.newBestMultiplier; // 2.0x for new best
            res.finalEpisodeReward = (m_episodeRawReward * res.rewardMultiplier) + res.terminalReward;
            res.saveToElite = true;
            res.fatalTrimSteps = 3; // imitate the winning path up to just before the fatal crash
        } else {
            m_stagnantEpisodes++;
            if (m_cfg.rollingPBStagnationEpisodes > 0 &&
                m_stagnantEpisodes >= m_cfg.rollingPBStagnationEpisodes) {
                // Rolling recovery soft floor: adjust session best slightly backwards
                // so the agent can regain learning momentum and earn 2x boosts again
                m_sessionBestX = std::max(0.0f, m_sessionBestX * m_cfg.rollingRecoveryFactor);
                m_stagnantEpisodes = 0;
            }
            res.rewardMultiplier = 0.0f; // no reward if it dies without PB
            res.finalEpisodeReward = res.terminalReward; // -1.0 collision penalty
            res.saveToElite = false;
            res.fatalTrimSteps = 0;
        }
        return res;
    }

    /// Resolves episode reward on level completion:
    /// - Full progress + completion reward, multiplied by 2x if it achieved a new best!
    EpisodeRewardResolution resolveCompletion() {
        EpisodeRewardResolution res;
        res.achievedNewBest = m_brokeRecordThisEpisode;
        m_stagnantEpisodes = 0;
        res.rewardMultiplier = m_brokeRecordThisEpisode ? m_cfg.newBestMultiplier : 1.0f;
        res.terminalReward = m_cfg.completionBonus;
        res.finalEpisodeReward = (m_episodeRawReward + m_cfg.completionBonus) * res.rewardMultiplier;
        res.saveToElite = true;
        res.fatalTrimSteps = 0;
        return res;
    }

    // Legacy accessors preserved for compatibility
    float computeDeathReward() const { return m_cfg.deathReward; }
    float computeCompletionReward() const { return m_cfg.completionBonus; }

    float sessionBestX() const { return m_sessionBestX; }
    bool brokeRecordThisEpisode() const { return m_brokeRecordThisEpisode; }
    float episodeRawReward() const { return m_episodeRawReward; }
    std::size_t stagnantEpisodes() const { return m_stagnantEpisodes; }
    RewardConfig& config() { return m_cfg; }
    const RewardConfig& config() const { return m_cfg; }

private:
    RewardConfig m_cfg;
    float m_prevX = 0.0f;
    float m_sessionBestX = 0.0f;
    float m_episodeStartBestX = 0.0f;
    int m_prevAction = 0;
    bool m_brokeRecordThisEpisode = false;
    float m_episodeRawReward = 0.0f;
    std::size_t m_stagnantEpisodes = 0;
};

} // namespace gdai::env
