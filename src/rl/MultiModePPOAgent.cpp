#include "MultiModePPOAgent.hpp"

#include <algorithm>
#include <fstream>

namespace gdai::rl {

namespace {
constexpr std::uint32_t kMasterMagic = 0x4744414D; // "GDAM" (GD-AI MultiMaster)
constexpr std::uint32_t kMasterVersion = 1;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
} // namespace

std::size_t MultiModePPOAgent::toIndex(env::Gamemode mode) {
    const auto idx = static_cast<std::size_t>(mode);
    return (idx < kModeCount) ? idx : 0;
}

PPOConfig MultiModePPOAgent::makeModeConfig(
    env::Gamemode mode,
    TrainingPreset preset,
    std::size_t obsDim,
    std::size_t obsNormDim
) {
    PPOConfig cfg = PPOConfig::fromPreset(preset, obsDim, obsNormDim);
    const auto idx = static_cast<std::uint32_t>(toIndex(mode));
    cfg.seed = 1337u + idx * 1009u;

    // Mode-specific hyperparameter tuning for distinct GD physics
    switch (mode) {
        case env::Gamemode::Ship:
        case env::Gamemode::Wave:
            // Continuous thrust / diagonal control benefits from slightly tighter clipping & lower terminal entropy
            cfg.clipEpsilon = std::min(cfg.clipEpsilon, 0.18f);
            cfg.entropyCoefEnd = std::min(cfg.entropyCoefEnd, 0.002f);
            break;
        case env::Gamemode::UFO:
        case env::Gamemode::Ball:
        case env::Gamemode::Spider:
        case env::Gamemode::Swing:
            // Discrete tap-timing modes benefit from slightly higher initial exploration
            cfg.entropyCoefStart = std::max(cfg.entropyCoefStart, 0.028f);
            break;
        case env::Gamemode::Robot:
            // Variable-duration hold jump benefits from smooth horizon
            cfg.gaeLambda = 0.96f;
            break;
        case env::Gamemode::Cube:
        default:
            break;
    }
    return cfg;
}

MultiModePPOAgent::MultiModePPOAgent(TrainingPreset preset, std::size_t obsDim, std::size_t obsNormDim) {
    m_agents.reserve(kModeCount);
    for (std::size_t i = 0; i < kModeCount; ++i) {
        const auto mode = static_cast<env::Gamemode>(i);
        m_agents.emplace_back(makeModeConfig(mode, preset, obsDim, obsNormDim));
        if (mode == env::Gamemode::Ship || mode == env::Gamemode::Wave) {
            m_agents.back().mutableNetwork().setActionBiases(0.9f, 0.1f, -1.0f);
        } else {
            m_agents.back().mutableNetwork().setActionBiases(1.3f, -0.5f, -2.4f);
        }
    }
}

bool MultiModePPOAgent::setActiveMode(env::Gamemode mode) {
    const auto safeMode = static_cast<env::Gamemode>(toIndex(mode));
    if (safeMode == m_activeMode) {
        return false;
    }
    m_activeMode = safeMode;
    ++m_hotSwapCount;
    return true;
}

PPOAgent& MultiModePPOAgent::agentFor(env::Gamemode mode) {
    return m_agents[toIndex(mode)];
}

const PPOAgent& MultiModePPOAgent::agentFor(env::Gamemode mode) const {
    return m_agents[toIndex(mode)];
}

ActionDecision MultiModePPOAgent::selectAction(env::Gamemode mode, const Vec& rawObs, bool training) {
    setActiveMode(mode);
    return agentFor(m_activeMode).selectAction(rawObs, training);
}

float MultiModePPOAgent::estimateValue(env::Gamemode mode, const Vec& rawObs) const {
    return agentFor(mode).estimateValue(rawObs);
}

void MultiModePPOAgent::stageTransition(
    env::Gamemode mode,
    const ActionDecision& decision,
    float rawReward,
    bool done,
    bool truncated,
    float bootstrapValue,
    std::uint64_t episodeId
) {
    agentFor(mode).stageTransition(decision, rawReward, done, truncated, bootstrapValue, episodeId);
}

std::vector<UpdateMetrics> MultiModePPOAgent::finalizeEpisode(
    std::uint64_t episodeId,
    float rewardMultiplier,
    float episodeScore,
    bool saveToElite,
    std::size_t fatalTrimSteps
) {
    std::vector<UpdateMetrics> updates;
    for (std::size_t i = 0; i < kModeCount; ++i) {
        // Only trim fatal crash frames from the gamemode where the player actually died;
        // earlier gamemodes in the same PB run survived their entire portal segment cleanly!
        const std::size_t modeTrim = (static_cast<env::Gamemode>(i) == m_activeMode) ? fatalTrimSteps : 0;
        if (auto upd = m_agents[i].finalizeEpisode(
                episodeId,
                rewardMultiplier,
                episodeScore,
                saveToElite,
                modeTrim,
                0.0f
            )) {
            updates.push_back(*upd);
        }
    }
    return updates;
}

std::optional<UpdateMetrics> MultiModePPOAgent::forceUpdateActive(float bootstrapValue) {
    return activeAgent().forceUpdate(bootstrapValue);
}

void MultiModePPOAgent::clearAllBuffers() {
    for (auto& a : m_agents) {
        a.clearBuffer();
    }
}

void MultiModePPOAgent::resetAll(std::uint32_t baseSeed) {
    for (std::size_t i = 0; i < kModeCount; ++i) {
        const auto mode = static_cast<env::Gamemode>(i);
        m_agents[i].reset(baseSeed + static_cast<std::uint32_t>(i) * 1009u);
        if (mode == env::Gamemode::Ship || mode == env::Gamemode::Wave) {
            m_agents[i].mutableNetwork().setActionBiases(0.9f, 0.1f, -1.0f);
        } else {
            m_agents[i].mutableNetwork().setActionBiases(1.3f, -0.5f, -2.4f);
        }
    }
    m_hotSwapCount = 0;
}

void MultiModePPOAgent::applyPresetAll(TrainingPreset preset) {
    for (std::size_t i = 0; i < kModeCount; ++i) {
        m_agents[i].applyPreset(preset);
    }
}

void MultiModePPOAgent::setLearningRateAll(float lr) {
    for (auto& a : m_agents) {
        a.setLearningRate(lr);
    }
}

void MultiModePPOAgent::setModeTrainingFrozen(env::Gamemode mode, bool frozen) {
    m_agents[toIndex(mode)].setTrainingFrozen(frozen);
}

bool MultiModePPOAgent::isModeTrainingFrozen(env::Gamemode mode) const {
    return m_agents[toIndex(mode)].isTrainingFrozen();
}

std::filesystem::path MultiModePPOAgent::masterBundlePathFor(
    const std::filesystem::path& dir,
    const std::string& prefix
) {
    return dir / (prefix + "_master.gdai");
}

bool MultiModePPOAgent::saveMasterBundle(const std::filesystem::path& bundlePath) const {
    std::error_code ec;
    if (bundlePath.has_parent_path()) {
        std::filesystem::create_directories(bundlePath.parent_path(), ec);
    }
    std::ofstream os(bundlePath, std::ios::binary | std::ios::trunc);
    if (!os) return false;

    const std::uint32_t modeCount = static_cast<std::uint32_t>(kModeCount);
    std::uint64_t checksum = kFnvOffset;

    writePod(os, kMasterMagic);
    writePod(os, kMasterVersion);
    writePod(os, modeCount);
    checksum = fnv1aUpdate(checksum, &kMasterMagic, sizeof(kMasterMagic));
    checksum = fnv1aUpdate(checksum, &kMasterVersion, sizeof(kMasterVersion));
    checksum = fnv1aUpdate(checksum, &modeCount, sizeof(modeCount));

    for (std::size_t i = 0; i < kModeCount; ++i) {
        const auto& agent = m_agents[i];
        const std::uint32_t modeId = static_cast<std::uint32_t>(i);
        const std::uint64_t steps = agent.totalSteps();
        const std::uint64_t updates = agent.updateCount();
        const std::uint8_t frozen = agent.isTrainingFrozen() ? 1 : 0;

        writePod(os, modeId);
        writePod(os, steps);
        writePod(os, updates);
        writePod(os, frozen);
        checksum = fnv1aUpdate(checksum, &modeId, sizeof(modeId));
        checksum = fnv1aUpdate(checksum, &steps, sizeof(steps));
        checksum = fnv1aUpdate(checksum, &updates, sizeof(updates));
        checksum = fnv1aUpdate(checksum, &frozen, sizeof(frozen));

        agent.network().save(os, checksum);
        agent.obsNorm().save(os, checksum);
    }

    writePod(os, checksum);
    return static_cast<bool>(os);
}

bool MultiModePPOAgent::loadMasterBundle(const std::filesystem::path& bundlePath) {
    std::ifstream is(bundlePath, std::ios::binary);
    if (!is) return false;

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t count = 0;
    if (!readPod(is, magic) || !readPod(is, version) || !readPod(is, count)) {
        return false;
    }
    if (magic != kMasterMagic || version != kMasterVersion || count != kModeCount) {
        return false;
    }

    std::uint64_t checksum = kFnvOffset;
    checksum = fnv1aUpdate(checksum, &magic, sizeof(magic));
    checksum = fnv1aUpdate(checksum, &version, sizeof(version));
    checksum = fnv1aUpdate(checksum, &count, sizeof(count));

    struct LoadedMode {
        std::uint32_t modeId = 0;
        std::uint64_t steps = 0;
        std::uint64_t updates = 0;
        std::uint8_t frozen = 0;
        std::unique_ptr<PolicyNetwork> net;
        std::unique_ptr<RunningMeanStd> norm;
    };
    std::vector<LoadedMode> loaded(kModeCount);

    for (std::size_t i = 0; i < kModeCount; ++i) {
        auto& lm = loaded[i];
        if (!readPod(is, lm.modeId) || !readPod(is, lm.steps) || !readPod(is, lm.updates) || !readPod(is, lm.frozen)) {
            return false;
        }
        checksum = fnv1aUpdate(checksum, &lm.modeId, sizeof(lm.modeId));
        checksum = fnv1aUpdate(checksum, &lm.steps, sizeof(lm.steps));
        checksum = fnv1aUpdate(checksum, &lm.updates, sizeof(lm.updates));
        checksum = fnv1aUpdate(checksum, &lm.frozen, sizeof(lm.frozen));

        const auto& cfg = m_agents[i].config();
        lm.net = std::make_unique<PolicyNetwork>(cfg.obsDim, cfg.hidden1, cfg.hidden2, cfg.seed);
        lm.norm = std::make_unique<RunningMeanStd>(cfg.obsDim, cfg.obsNormDim);

        if (!lm.net->load(is, checksum) || !lm.norm->load(is, checksum)) {
            return false;
        }
    }

    std::uint64_t storedChecksum = 0;
    if (!readPod(is, storedChecksum) || storedChecksum != checksum) {
        return false;
    }

    // Checksum verified: apply to all agents atomically
    for (std::size_t i = 0; i < kModeCount; ++i) {
        auto& agent = m_agents[i];
        auto& lm = loaded[i];
        agent.mutableNetwork() = std::move(*lm.net);
        agent.mutableObsNorm() = std::move(*lm.norm);
        agent.setStepsAndUpdates(lm.steps, lm.updates);
        agent.setTrainingFrozen(lm.frozen != 0);
    }
    return true;
}

std::filesystem::path MultiModePPOAgent::modelPathFor(
    const std::filesystem::path& dir,
    const std::string& prefix,
    env::Gamemode mode
) {
    return dir / (prefix + "_" + env::gamemodeSlug(mode) + ".gdai");
}

std::size_t MultiModePPOAgent::saveAllModels(
    const std::filesystem::path& dir,
    const std::string& prefix,
    bool saveUntrained
) const {
    std::size_t saved = 0;
    for (std::size_t i = 0; i < kModeCount; ++i) {
        const auto mode = static_cast<env::Gamemode>(i);
        if (!saveUntrained && m_agents[i].totalSteps() == 0 && m_agents[i].updateCount() == 0) {
            continue;
        }
        const auto path = modelPathFor(dir, prefix, mode);
        if (m_agents[i].saveModel(path)) {
            ++saved;
        }
    }
    return saved;
}

std::size_t MultiModePPOAgent::loadAllModels(
    const std::filesystem::path& dir,
    const std::string& prefix
) {
    std::size_t loaded = 0;
    for (std::size_t i = 0; i < kModeCount; ++i) {
        const auto mode = static_cast<env::Gamemode>(i);
        const auto path = modelPathFor(dir, prefix, mode);
        if (std::filesystem::exists(path) && m_agents[i].loadModel(path)) {
            ++loaded;
        }
    }
    return loaded;
}

std::uint64_t MultiModePPOAgent::totalStepsAll() const {
    std::uint64_t sum = 0;
    for (const auto& a : m_agents) sum += a.totalSteps();
    return sum;
}

std::uint64_t MultiModePPOAgent::totalUpdatesAll() const {
    std::uint64_t sum = 0;
    for (const auto& a : m_agents) sum += a.updateCount();
    return sum;
}

std::size_t MultiModePPOAgent::totalParametersAll() const {
    std::size_t sum = 0;
    for (const auto& a : m_agents) sum += a.network().parameterCount();
    return sum;
}

std::size_t MultiModePPOAgent::trainedModeCount() const {
    std::size_t count = 0;
    for (const auto& a : m_agents) {
        if (a.updateCount() > 0) ++count;
    }
    return count;
}

} // namespace gdai::rl
