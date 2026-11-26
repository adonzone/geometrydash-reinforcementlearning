#pragma once

#include "Observation.hpp"
#include "../rl/PPOAgent.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

namespace gdai::env {

struct EpisodeRecord {
    std::uint64_t episodeIndex = 0;
    float progressPercent = 0.0f;
    float totalReward = 0.0f;
    std::uint32_t steps = 0;
    bool completed = false;
    Gamemode deathMode = Gamemode::Cube;
};

struct GamemodeStats {
    std::uint64_t steps = 0;
    std::uint64_t deaths = 0;
};

/// Tracks live telemetry, ring-buffered history for HUD/Dashboard graphs, and CSV export.
class StatsTracker {
public:
    static constexpr std::size_t kMaxHistory = 120;

    void resetAll();
    void onLevelChanged(int levelId, std::string levelName);

    void onStep(
        float progressPercent,
        float stepReward,
        int action,
        float holdProb,
        float valueEstimate,
        Gamemode mode
    );

    void onEpisodeEnd(float finalPercent, bool completed, Gamemode mode);
    void onEpisodeEnd(float finalPercent, bool completed, Gamemode mode, float finalizedEpisodeReward);
    void onPPOUpdate(const rl::UpdateMetrics& metrics);

    // Rolling aggregates
    float rollingAvgPercent(std::size_t window = 20) const;
    float rollingAvgReward(std::size_t window = 20) const;
    float rollingWinRate(std::size_t window = 50) const;
    float rollingAvgSteps(std::size_t window = 20) const;

    // Chart data series (oldest -> newest)
    std::vector<float> percentSeries() const;
    std::vector<float> rollingPercentSeries(std::size_t window = 10) const;
    std::vector<float> rewardSeries() const;
    std::vector<float> entropySeries() const;
    std::vector<float> policyLossSeries() const;
    std::vector<float> valueLossSeries() const;

    bool exportCSV(const std::filesystem::path& path) const;

    // Live getters
    int levelId() const { return m_levelId; }
    const std::string& levelName() const { return m_levelName; }

    std::uint64_t totalEpisodes() const { return m_totalEpisodes; }
    std::uint64_t totalSteps() const { return m_totalSteps; }
    std::uint64_t totalCompletions() const { return m_totalCompletions; }
    std::uint64_t totalUpdates() const { return m_totalUpdates; }

    float currentPercent() const { return m_currentPercent; }
    float bestPercent() const { return m_bestPercent; }
    float bestReward() const { return m_bestReward; }
    float currentEpisodeReward() const { return m_currentEpisodeReward; }
    std::uint32_t currentEpisodeSteps() const { return m_currentEpisodeSteps; }

    float lastEpisodePercent() const { return m_lastEpisodePercent; }
    float lastEpisodeReward() const { return m_lastEpisodeReward; }

    int lastAction() const { return m_lastAction; }
    float lastHoldProb() const { return m_lastHoldProb; }
    float lastValueEstimate() const { return m_lastValueEstimate; }
    Gamemode currentGamemode() const { return m_currentGamemode; }

    const rl::UpdateMetrics& lastUpdate() const { return m_lastUpdate; }
    const GamemodeStats& gamemodeStats(Gamemode mode) const {
        return m_modeStats[static_cast<std::size_t>(mode)];
    }

private:
    int m_levelId = 0;
    std::string m_levelName = "None";

    std::uint64_t m_totalEpisodes = 0;
    std::uint64_t m_totalSteps = 0;
    std::uint64_t m_totalCompletions = 0;
    std::uint64_t m_totalUpdates = 0;

    float m_currentPercent = 0.0f;
    float m_bestPercent = 0.0f;
    float m_bestReward = -1e9f;
    float m_currentEpisodeReward = 0.0f;
    std::uint32_t m_currentEpisodeSteps = 0;

    float m_lastEpisodePercent = 0.0f;
    float m_lastEpisodeReward = 0.0f;

    int m_lastAction = 0;
    float m_lastHoldProb = 0.5f;
    float m_lastValueEstimate = 0.0f;
    Gamemode m_currentGamemode = Gamemode::Cube;

    rl::UpdateMetrics m_lastUpdate;

    std::deque<EpisodeRecord> m_episodeHistory;
    std::deque<rl::UpdateMetrics> m_updateHistory;
    std::array<GamemodeStats, static_cast<std::size_t>(Gamemode::Count)> m_modeStats{};
};

} // namespace gdai::env
