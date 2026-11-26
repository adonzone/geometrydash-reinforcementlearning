#include "StatsTracker.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>

namespace gdai::env {

void StatsTracker::resetAll() {
    m_totalEpisodes = 0;
    m_totalSteps = 0;
    m_totalCompletions = 0;
    m_totalUpdates = 0;
    m_currentPercent = 0.0f;
    m_bestPercent = 0.0f;
    m_bestReward = -1e9f;
    m_currentEpisodeReward = 0.0f;
    m_currentEpisodeSteps = 0;
    m_lastEpisodePercent = 0.0f;
    m_lastEpisodeReward = 0.0f;
    m_lastAction = 0;
    m_lastHoldProb = 0.5f;
    m_lastValueEstimate = 0.0f;
    m_currentGamemode = Gamemode::Cube;
    m_lastUpdate = rl::UpdateMetrics{};
    m_episodeHistory.clear();
    m_updateHistory.clear();
    m_modeStats.fill(GamemodeStats{});
}

void StatsTracker::onLevelChanged(int levelId, std::string levelName) {
    m_levelId = levelId;
    m_levelName = std::move(levelName);
    m_currentPercent = 0.0f;
    m_currentEpisodeReward = 0.0f;
    m_currentEpisodeSteps = 0;
}

void StatsTracker::onStep(
    float progressPercent,
    float stepReward,
    int action,
    float holdProb,
    float valueEstimate,
    Gamemode mode
) {
    m_currentPercent = std::clamp(progressPercent, 0.0f, 100.0f);
    if (m_currentPercent > m_bestPercent) {
        m_bestPercent = m_currentPercent;
    }
    m_currentEpisodeReward += stepReward;
    ++m_currentEpisodeSteps;
    ++m_totalSteps;

    m_lastAction = action;
    m_lastHoldProb = holdProb;
    m_lastValueEstimate = valueEstimate;
    m_currentGamemode = mode;
    ++m_modeStats[static_cast<std::size_t>(mode)].steps;
}

void StatsTracker::onEpisodeEnd(float finalPercent, bool completed, Gamemode mode) {
    onEpisodeEnd(finalPercent, completed, mode, m_currentEpisodeReward);
}

void StatsTracker::onEpisodeEnd(float finalPercent, bool completed, Gamemode mode, float finalizedEpisodeReward) {
    const float clampedPct = completed ? 100.0f : std::clamp(finalPercent, 0.0f, 100.0f);
    if (clampedPct > m_bestPercent) {
        m_bestPercent = clampedPct;
    }
    if (finalizedEpisodeReward > m_bestReward) {
        m_bestReward = finalizedEpisodeReward;
    }

    ++m_totalEpisodes;
    if (completed) {
        ++m_totalCompletions;
    } else {
        ++m_modeStats[static_cast<std::size_t>(mode)].deaths;
    }

    m_lastEpisodePercent = clampedPct;
    m_lastEpisodeReward = finalizedEpisodeReward;

    EpisodeRecord rec;
    rec.episodeIndex = m_totalEpisodes;
    rec.progressPercent = clampedPct;
    rec.totalReward = finalizedEpisodeReward;
    rec.steps = m_currentEpisodeSteps;
    rec.completed = completed;
    rec.deathMode = mode;

    m_episodeHistory.push_back(rec);
    if (m_episodeHistory.size() > kMaxHistory) {
        m_episodeHistory.pop_front();
    }

    m_currentPercent = 0.0f;
    m_currentEpisodeReward = 0.0f;
    m_currentEpisodeSteps = 0;
}

void StatsTracker::onPPOUpdate(const rl::UpdateMetrics& metrics) {
    ++m_totalUpdates;
    m_lastUpdate = metrics;
    m_updateHistory.push_back(metrics);
    if (m_updateHistory.size() > kMaxHistory) {
        m_updateHistory.pop_front();
    }
}

float StatsTracker::rollingAvgPercent(std::size_t window) const {
    if (m_episodeHistory.empty()) return 0.0f;
    const std::size_t count = std::min(window, m_episodeHistory.size());
    double sum = 0.0;
    for (std::size_t i = m_episodeHistory.size() - count; i < m_episodeHistory.size(); ++i) {
        sum += m_episodeHistory[i].progressPercent;
    }
    return static_cast<float>(sum / static_cast<double>(count));
}

float StatsTracker::rollingAvgReward(std::size_t window) const {
    if (m_episodeHistory.empty()) return 0.0f;
    const std::size_t count = std::min(window, m_episodeHistory.size());
    double sum = 0.0;
    for (std::size_t i = m_episodeHistory.size() - count; i < m_episodeHistory.size(); ++i) {
        sum += m_episodeHistory[i].totalReward;
    }
    return static_cast<float>(sum / static_cast<double>(count));
}

float StatsTracker::rollingWinRate(std::size_t window) const {
    if (m_episodeHistory.empty()) return 0.0f;
    const std::size_t count = std::min(window, m_episodeHistory.size());
    std::size_t wins = 0;
    for (std::size_t i = m_episodeHistory.size() - count; i < m_episodeHistory.size(); ++i) {
        if (m_episodeHistory[i].completed) ++wins;
    }
    return 100.0f * static_cast<float>(wins) / static_cast<float>(count);
}

float StatsTracker::rollingAvgSteps(std::size_t window) const {
    if (m_episodeHistory.empty()) return 0.0f;
    const std::size_t count = std::min(window, m_episodeHistory.size());
    double sum = 0.0;
    for (std::size_t i = m_episodeHistory.size() - count; i < m_episodeHistory.size(); ++i) {
        sum += m_episodeHistory[i].steps;
    }
    return static_cast<float>(sum / static_cast<double>(count));
}

std::vector<float> StatsTracker::percentSeries() const {
    std::vector<float> out;
    out.reserve(m_episodeHistory.size());
    for (const auto& ep : m_episodeHistory) out.push_back(ep.progressPercent);
    return out;
}

std::vector<float> StatsTracker::rollingPercentSeries(std::size_t window) const {
    std::vector<float> out;
    out.reserve(m_episodeHistory.size());
    double running = 0.0;
    for (std::size_t i = 0; i < m_episodeHistory.size(); ++i) {
        running += m_episodeHistory[i].progressPercent;
        if (i >= window) {
            running -= m_episodeHistory[i - window].progressPercent;
        }
        const std::size_t denom = std::min(i + 1, window);
        out.push_back(static_cast<float>(running / static_cast<double>(denom)));
    }
    return out;
}

std::vector<float> StatsTracker::rewardSeries() const {
    std::vector<float> out;
    out.reserve(m_episodeHistory.size());
    for (const auto& ep : m_episodeHistory) out.push_back(ep.totalReward);
    return out;
}

std::vector<float> StatsTracker::entropySeries() const {
    std::vector<float> out;
    out.reserve(m_updateHistory.size());
    for (const auto& u : m_updateHistory) out.push_back(u.entropy);
    return out;
}

std::vector<float> StatsTracker::policyLossSeries() const {
    std::vector<float> out;
    out.reserve(m_updateHistory.size());
    for (const auto& u : m_updateHistory) out.push_back(u.policyLoss);
    return out;
}

std::vector<float> StatsTracker::valueLossSeries() const {
    std::vector<float> out;
    out.reserve(m_updateHistory.size());
    for (const auto& u : m_updateHistory) out.push_back(u.valueLoss);
    return out;
}

bool StatsTracker::exportCSV(const std::filesystem::path& path) const {
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream os(path, std::ios::trunc);
    if (!os) return false;

    os << "episode,progress_percent,total_reward,steps,completed,death_mode\n";
    os << std::fixed << std::setprecision(4);
    for (const auto& ep : m_episodeHistory) {
        os << ep.episodeIndex << ','
           << ep.progressPercent << ','
           << ep.totalReward << ','
           << ep.steps << ','
           << (ep.completed ? 1 : 0) << ','
           << gamemodeName(ep.deathMode) << '\n';
    }
    return static_cast<bool>(os);
}

} // namespace gdai::env
