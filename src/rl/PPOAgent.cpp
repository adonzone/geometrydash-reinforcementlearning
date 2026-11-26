#include "PPOAgent.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace gdai::rl {

namespace {
constexpr std::uint32_t kModelMagic = 0x49414447; // "GDAI"
constexpr std::uint32_t kModelVersion = 2;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::size_t kMaxPendingEpisodeSteps = 4096;
} // namespace

PPOConfig PPOConfig::fromPreset(TrainingPreset preset, std::size_t obsDim, std::size_t obsNormDim) {
    PPOConfig cfg;
    cfg.obsDim = obsDim;
    cfg.obsNormDim = obsNormDim;
    switch (preset) {
        case TrainingPreset::FastLearner:
            cfg.rolloutHorizon = 256;
            cfg.minibatchSize = 64;
            cfg.epochs = 4;
            cfg.learningRate = 5e-4f;
            cfg.minLearningRate = 1e-4f;
            cfg.entropyCoefStart = 0.03f;
            cfg.entropyCoefEnd = 0.005f;
            cfg.clipEpsilon = 0.22f;
            cfg.silCoef = 0.20f;
            cfg.silBatches = 2;
            break;
        case TrainingPreset::BalancedMaster:
            cfg.rolloutHorizon = 512;
            cfg.minibatchSize = 64;
            cfg.epochs = 4;
            cfg.learningRate = 3e-4f;
            cfg.minLearningRate = 3e-5f;
            cfg.entropyCoefStart = 0.025f;
            cfg.entropyCoefEnd = 0.003f;
            cfg.clipEpsilon = 0.2f;
            cfg.silCoef = 0.16f;
            cfg.silBatches = 2;
            break;
        case TrainingPreset::FineTunePolish:
            cfg.rolloutHorizon = 1024;
            cfg.minibatchSize = 128;
            cfg.epochs = 3;
            cfg.learningRate = 1e-4f;
            cfg.minLearningRate = 1e-5f;
            cfg.entropyCoefStart = 0.008f;
            cfg.entropyCoefEnd = 0.001f;
            cfg.clipEpsilon = 0.12f;
            cfg.targetKL = 0.015f;
            cfg.silCoef = 0.10f;
            cfg.silBatches = 1;
            break;
    }
    return cfg;
}

PPOAgent::PPOAgent(PPOConfig config)
    : m_config(config)
    , m_net(config.obsDim, config.hidden1, config.hidden2, config.seed)
    , m_obsNorm(config.obsDim, config.obsNormDim)
    , m_buffer(config.rolloutHorizon)
    , m_rng(config.seed + 9973u) {
    m_optim.registerParams(m_net.paramSlots());
    m_pendingEpisode.reserve(512);
    m_eliteSamples.reserve(config.maxEliteSamples);
}

ActionDecision PPOAgent::selectAction(const Vec& rawObs, bool training) {
    if (training && m_config.normalizeObs) {
        m_obsNorm.update(rawObs);
    }
    Vec normObs = m_config.normalizeObs ? m_obsNorm.normalize(rawObs) : rawObs;
    const ForwardCache cache = m_net.forward(normObs);

    const float p0 = std::clamp(cache.probs[0], 1e-6f, 1.0f - 1e-6f);
    const float p1 = std::clamp(cache.probs[1], 1e-6f, 1.0f - 1e-6f);
    const float p2 = std::clamp(cache.probs[2], 1e-6f, 1.0f - 1e-6f);
    const float sumP = p0 + p1 + p2;
    const float normP0 = p0 / sumP;
    const float normP1 = p1 / sumP;
    const float normP2 = p2 / sumP;

    int action = 0;
    if (training) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        const float sample = u(m_rng);
        if (sample < normP0) {
            action = 0; // RELEASE
        } else if (sample < normP0 + normP1) {
            action = 1; // CLICK (TAP)
        } else {
            action = 2; // HOLD
        }
    } else {
        // Deterministic argmax policy for evaluation
        if (normP0 >= normP1 && normP0 >= normP2) {
            action = 0; // RELEASE
        } else if (normP1 >= normP2) {
            action = 1; // CLICK
        } else {
            action = 2; // HOLD
        }
    }

    const float chosenProb = (action == 0) ? normP0 : (action == 1 ? normP1 : normP2);
    const float logProb = std::log(std::max(1e-7f, chosenProb));

    ActionDecision out;
    out.action = action;
    out.releaseProb = normP0;
    out.clickProb = normP1;
    out.holdProb = normP2;
    out.logProb = logProb;
    out.value = cache.value;
    out.normObs = std::move(normObs);
    return out;
}

float PPOAgent::estimateValue(const Vec& rawObs) const {
    if (rawObs.size() != m_config.obsDim) return 0.0f;
    const Vec normObs = m_config.normalizeObs ? m_obsNorm.normalize(rawObs) : rawObs;
    return m_net.forward(normObs).value;
}

std::optional<UpdateMetrics> PPOAgent::recordTransition(
    const ActionDecision& decision,
    float reward,
    bool done,
    const Vec* nextRawObs
) {
    Transition t;
    t.normObs = decision.normObs;
    t.action = decision.action;
    t.logProb = decision.logProb;
    t.value = decision.value;
    t.reward = reward;
    t.done = done;

    m_buffer.add(std::move(t));
    ++m_totalSteps;

    if (!m_buffer.full()) {
        return std::nullopt;
    }

    float bootstrapValue = 0.0f;
    if (!done && nextRawObs != nullptr && nextRawObs->size() == m_config.obsDim) {
        const Vec nextNorm = m_config.normalizeObs ? m_obsNorm.normalize(*nextRawObs) : *nextRawObs;
        bootstrapValue = m_net.forward(nextNorm).value;
    }
    return runPPOUpdate(bootstrapValue);
}

void PPOAgent::stageTransition(
    const ActionDecision& decision,
    float rawReward,
    bool done,
    bool truncated,
    float bootstrapValue,
    std::uint64_t episodeId
) {
    Transition t;
    t.normObs = decision.normObs;
    t.action = decision.action;
    t.logProb = decision.logProb;
    t.value = decision.value;
    t.reward = rawReward;
    t.done = done;
    t.truncated = truncated;
    t.bootstrapValue = bootstrapValue;
    t.episodeId = episodeId;

    if (m_trainingFrozen) {
        ++m_totalSteps;
        return;
    }

    if (m_pendingEpisode.size() >= kMaxPendingEpisodeSteps) {
        // Safety cap for extraordinarily long survival runs: drop oldest staged element
        m_pendingEpisode.erase(m_pendingEpisode.begin());
    }
    m_pendingEpisode.push_back(std::move(t));
    ++m_totalSteps;
}

std::optional<UpdateMetrics> PPOAgent::finalizeEpisode(
    std::uint64_t episodeId,
    float rewardMultiplier,
    float episodeScore,
    bool saveToElite,
    std::size_t fatalTrimSteps,
    float bootstrapValue
) {
    if (m_pendingEpisode.empty()) {
        return std::nullopt;
    }

    std::vector<Transition> epTransitions;
    epTransitions.reserve(m_pendingEpisode.size());

    std::vector<Transition> remaining;
    for (auto& tr : m_pendingEpisode) {
        if (tr.episodeId == episodeId) {
            if (rewardMultiplier == 0.0f) {
                // Zero out all progress rewards on death without a new best,
                // while preserving the negative terminal collision penalty (-1.0)
                if (!tr.done) {
                    tr.reward = 0.0f;
                }
            } else {
                tr.reward *= rewardMultiplier;
            }
            epTransitions.push_back(std::move(tr));
        } else {
            remaining.push_back(std::move(tr));
        }
    }
    m_pendingEpisode = std::move(remaining);

    if (epTransitions.empty() || m_trainingFrozen) {
        return std::nullopt;
    }

    if (saveToElite && m_config.maxEliteSamples > 0) {
        storeEliteTrajectory(epTransitions, episodeScore, fatalTrimSteps);
    }

    for (auto& tr : epTransitions) {
        m_buffer.add(std::move(tr));
    }

    if (m_buffer.full()) {
        return runPPOUpdate(bootstrapValue);
    }
    return std::nullopt;
}

std::optional<UpdateMetrics> PPOAgent::forceUpdate(float bootstrapValue) {
    if (m_buffer.size() < std::max<std::size_t>(16, m_config.minibatchSize / 2)) {
        return std::nullopt;
    }
    return runPPOUpdate(bootstrapValue);
}

void PPOAgent::clearBuffer() {
    m_buffer.clear();
    m_pendingEpisode.clear();
}

void PPOAgent::reset(std::uint32_t newSeed) {
    if (newSeed != 0) {
        m_config.seed = newSeed;
    }
    m_net.resetWeights(m_config.seed);
    m_optim.resetState();
    m_obsNorm.reset(m_config.obsDim, m_config.obsNormDim);
    m_buffer.clear();
    m_pendingEpisode.clear();
    m_eliteSamples.clear();
    m_rng.seed(m_config.seed + 9973u);
    m_totalSteps = 0;
    m_updateCount = 0;
    m_lastMetrics = UpdateMetrics{};
}

void PPOAgent::applyPreset(TrainingPreset preset) {
    const auto p = PPOConfig::fromPreset(preset, m_config.obsDim, m_config.obsNormDim);
    m_config.rolloutHorizon = p.rolloutHorizon;
    m_config.minibatchSize = p.minibatchSize;
    m_config.epochs = p.epochs;
    m_config.learningRate = p.learningRate;
    m_config.minLearningRate = p.minLearningRate;
    m_config.entropyCoefStart = p.entropyCoefStart;
    m_config.entropyCoefEnd = p.entropyCoefEnd;
    m_config.clipEpsilon = p.clipEpsilon;
    m_config.targetKL = p.targetKL;
    m_config.silCoef = p.silCoef;
    m_config.silBatches = p.silBatches;
    m_buffer.resize(m_config.rolloutHorizon);
}

void PPOAgent::setLearningRate(float lr) {
    m_config.learningRate = std::clamp(lr, 1e-5f, 5e-3f);
}

float PPOAgent::currentLearningRate() const {
    if (m_config.lrAnnealUpdates == 0) return m_config.learningRate;
    const float progress = std::min(
        1.0f,
        static_cast<float>(m_updateCount) / static_cast<float>(m_config.lrAnnealUpdates)
    );
    const float cosFactor = 0.5f * (1.0f + std::cos(3.1415926535f * progress));
    return m_config.minLearningRate + (m_config.learningRate - m_config.minLearningRate) * cosFactor;
}

float PPOAgent::currentEntropyCoef() const {
    if (m_config.entropyAnnealUpdates == 0) return m_config.entropyCoefStart;
    const float progress = std::min(
        1.0f,
        static_cast<float>(m_updateCount) / static_cast<float>(m_config.entropyAnnealUpdates)
    );
    return m_config.entropyCoefStart + (m_config.entropyCoefEnd - m_config.entropyCoefStart) * progress;
}

void PPOAgent::storeEliteTrajectory(
    const std::vector<Transition>& episode,
    float episodeScore,
    std::size_t fatalTrimSteps
) {
    const std::size_t n = episode.size();
    if (n <= fatalTrimSteps) return;

    // Compute exact discounted Monte-Carlo returns along the winning trajectory
    std::vector<float> returns(n, 0.0f);
    float runningRet = 0.0f;
    for (std::size_t rev = 0; rev < n; ++rev) {
        const std::size_t t = n - 1 - rev;
        const auto& tr = episode[t];
        if (tr.done) {
            runningRet = tr.reward;
        } else if (tr.truncated) {
            runningRet = tr.reward + m_config.gamma * tr.bootstrapValue;
        } else {
            runningRet = tr.reward + m_config.gamma * runningRet;
        }
        returns[t] = runningRet;
    }

    const std::size_t keepCount = n - fatalTrimSteps;
    for (std::size_t t = 0; t < keepCount; ++t) {
        if (returns[t] <= 0.0f) continue;
        EliteSample sample;
        sample.normObs = episode[t].normObs;
        sample.action = episode[t].action;
        sample.discountedReturn = returns[t];
        sample.episodeScore = episodeScore;
        m_eliteSamples.push_back(std::move(sample));
    }

    // Keep the highest-scoring / freshest elite transitions within capacity
    if (m_eliteSamples.size() > m_config.maxEliteSamples) {
        std::stable_sort(
            m_eliteSamples.begin(),
            m_eliteSamples.end(),
            [](const EliteSample& a, const EliteSample& b) {
                return a.episodeScore > b.episodeScore;
            }
        );
        m_eliteSamples.resize(m_config.maxEliteSamples);
    }
}

void PPOAgent::runSelfImitationPass(float lr) {
    if (m_eliteSamples.empty() || m_config.silCoef <= 0.0f || m_config.silBatches == 0) {
        return;
    }

    const std::size_t eliteN = m_eliteSamples.size();
    const std::size_t batchLen = std::min(m_config.minibatchSize, eliteN);
    if (batchLen < 4) return;

    std::uniform_int_distribution<std::size_t> dist(0, eliteN - 1);
    const float invBatch = 1.0f / static_cast<float>(batchLen);
    const float silCoef = m_config.silCoef;
    const float vCoef = m_config.valueLossCoef;

    for (std::size_t b = 0; b < m_config.silBatches; ++b) {
        m_optim.zeroGrad();
        std::size_t activeCount = 0;

        for (std::size_t k = 0; k < batchLen; ++k) {
            const auto& sample = m_eliteSamples[dist(m_rng)];
            const ForwardCache cache = m_net.forward(sample.normObs);

            // SIL only imitates when elite return strictly beats current critic estimate
            const float rawAdv = sample.discountedReturn - cache.value;
            if (rawAdv <= 1e-4f) continue;

            const float adv = std::min(rawAdv, 10.0f);
            const int a = std::clamp(sample.action, 0, 2);

            float dLogits[3];
            for (int i = 0; i < 3; ++i) {
                const float pi = std::clamp(cache.probs[i], 1e-6f, 1.0f - 1e-6f);
                const float dLogProb_dLogit = ((i == a) ? 1.0f : 0.0f) - pi;
                dLogits[i] = (-silCoef * adv * dLogProb_dLogit) * invBatch;
            }

            // Pull V(s) upward toward the proven elite return (Huber-clamped)
            const float dValue = (-vCoef * adv) * invBatch;

            m_net.backwardAccumulate(sample.normObs, cache, dLogits[0], dLogits[1], dLogits[2], dValue);
            ++activeCount;
        }

        if (activeCount > 0) {
            m_optim.step(lr * 0.65f, m_config.maxGradNorm);
        }
    }
}

UpdateMetrics PPOAgent::runPPOUpdate(float bootstrapValue) {
    const std::size_t n = m_buffer.size();
    m_buffer.computeGAE(bootstrapValue, m_config.gamma, m_config.gaeLambda);

    // Compute explained variance before update: 1 - Var(y - y_pred) / Var(y)
    double retMean = 0.0;
    double errMean = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double r = m_buffer.ret(i);
        const double e = r - m_buffer.item(i).value;
        retMean += r;
        errMean += e;
    }
    retMean /= static_cast<double>(n);
    errMean /= static_cast<double>(n);

    double retVar = 0.0;
    double errVar = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double r = m_buffer.ret(i);
        const double e = r - m_buffer.item(i).value;
        retVar += (r - retMean) * (r - retMean);
        errVar += (e - errMean) * (e - errMean);
    }
    const float explainedVar = (retVar > 1e-8) ? static_cast<float>(1.0 - errVar / retVar) : 0.0f;

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    const float lr = currentLearningRate();
    const float entCoef = currentEntropyCoef();
    const float clipEps = m_config.clipEpsilon;
    const float vClipEps = m_config.valueClipEpsilon;
    const float vCoef = m_config.valueLossCoef;

    double sumPolicyLoss = 0.0;
    double sumValueLoss = 0.0;
    double sumEntropy = 0.0;
    double sumKL = 0.0;
    double sumClipped = 0.0;
    double sumGradNorm = 0.0;
    std::size_t totalBatches = 0;
    std::size_t totalSamples = 0;
    bool earlyStopped = false;

    for (std::size_t epoch = 0; epoch < m_config.epochs && !earlyStopped; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), m_rng);

        for (std::size_t start = 0; start < n; start += m_config.minibatchSize) {
            const std::size_t end = std::min(start + m_config.minibatchSize, n);
            const std::size_t batchLen = end - start;
            if (batchLen < 4) continue;

            // Normalize advantages per minibatch
            double advMean = 0.0;
            for (std::size_t k = start; k < end; ++k) {
                advMean += m_buffer.advantage(indices[k]);
            }
            advMean /= static_cast<double>(batchLen);

            double advVar = 0.0;
            for (std::size_t k = start; k < end; ++k) {
                const double d = m_buffer.advantage(indices[k]) - advMean;
                advVar += d * d;
            }
            const float advInvStd = 1.0f / static_cast<float>(std::sqrt(advVar / static_cast<double>(batchLen)) + 1e-8);

            m_optim.zeroGrad();
            const float invBatch = 1.0f / static_cast<float>(batchLen);
            double batchKL = 0.0;

            for (std::size_t k = start; k < end; ++k) {
                const std::size_t idx = indices[k];
                const Transition& tr = m_buffer.item(idx);
                const float adv = (m_buffer.advantage(idx) - static_cast<float>(advMean)) * advInvStd;
                const float targetRet = m_buffer.ret(idx);

                const ForwardCache cache = m_net.forward(tr.normObs);
                const auto& probs = cache.probs;
                const int a = std::clamp(tr.action, 0, 2);
                const float p_chosen = std::clamp(probs[a], 1e-6f, 1.0f - 1e-6f);

                const float newLogProb = std::log(p_chosen);
                const float logRatio = std::clamp(newLogProb - tr.logProb, -20.0f, 20.0f);
                const float ratio = std::exp(logRatio);

                // Schulman unbiased KL estimator: (ratio - 1) - logRatio
                const float approxKL = (ratio - 1.0f) - logRatio;
                batchKL += approxKL;

                // Clipped PPO surrogate objective
                const float surr1 = ratio * adv;
                const float clippedRatio = std::clamp(ratio, 1.0f - clipEps, 1.0f + clipEps);
                const float surr2 = clippedRatio * adv;
                const bool useUnclipped = (surr1 <= surr2);
                const float policyLoss = -std::min(surr1, surr2);
                if (std::abs(ratio - 1.0f) > clipEps) {
                    sumClipped += 1.0;
                }

                const float dL_dLogProb = useUnclipped ? (-ratio * adv) : 0.0f;

                // 3-class categorical entropy: H(p) = - sum_j p_j * log(p_j)
                float entropy = 0.0f;
                for (int j = 0; j < 3; ++j) {
                    const float pj = std::clamp(probs[j], 1e-6f, 1.0f - 1e-6f);
                    entropy -= pj * std::log(pj);
                }

                // Exact softmax derivative with entropy regularization:
                // dLogProb / dLogit_i = (i == a) - p_i
                // dEntropy / dLogit_i = - p_i * (log(p_i) + entropy)
                float dLogits[3];
                for (int i = 0; i < 3; ++i) {
                    const float pi = std::clamp(probs[i], 1e-6f, 1.0f - 1e-6f);
                    const float dLogProb_dLogit = ((i == a) ? 1.0f : 0.0f) - pi;
                    const float dEnt_dLogit = -pi * (std::log(pi) + entropy);
                    dLogits[i] = (dL_dLogProb * dLogProb_dLogit - entCoef * dEnt_dLogit) * invBatch;
                }

                // Huber-stabilized clipped value loss to stay robust under 0x death vs 2x PB spikes
                constexpr float kHuberDelta = 10.0f;
                const float vDiff = cache.value - targetRet;
                const float vLossUnclipped = 0.5f * vDiff * vDiff;
                const float vClipped = tr.value + std::clamp(cache.value - tr.value, -vClipEps, vClipEps);
                const float vDiffClipped = vClipped - targetRet;
                const float vLossClipped = 0.5f * vDiffClipped * vDiffClipped;

                float vLoss = 0.0f;
                float dValue = 0.0f;
                if (vLossUnclipped >= vLossClipped) {
                    vLoss = vLossUnclipped;
                    dValue = vCoef * std::clamp(vDiff, -kHuberDelta, kHuberDelta) * invBatch;
                } else {
                    vLoss = vLossClipped;
                    const bool withinClip = std::abs(cache.value - tr.value) <= vClipEps;
                    dValue = withinClip ? (vCoef * std::clamp(vDiffClipped, -kHuberDelta, kHuberDelta) * invBatch) : 0.0f;
                }

                m_net.backwardAccumulate(tr.normObs, cache, dLogits[0], dLogits[1], dLogits[2], dValue);

                sumPolicyLoss += policyLoss;
                sumValueLoss += vLoss;
                sumEntropy += entropy;
                sumKL += approxKL;
                ++totalSamples;
            }

            const float gradNorm = m_optim.step(lr, m_config.maxGradNorm);
            sumGradNorm += gradNorm;
            ++totalBatches;

            const float meanBatchKL = static_cast<float>(batchKL / static_cast<double>(batchLen));
            if (m_config.targetKL > 0.0f && meanBatchKL > 1.5f * m_config.targetKL) {
                earlyStopped = true;
                break;
            }
        }
    }

    // Reinforce proven 2x new-best & completion runs via Self-Imitation Learning
    runSelfImitationPass(lr);

    m_buffer.clear();
    ++m_updateCount;

    UpdateMetrics metrics;
    if (totalSamples > 0) {
        metrics.policyLoss = static_cast<float>(sumPolicyLoss / static_cast<double>(totalSamples));
        metrics.valueLoss = static_cast<float>(sumValueLoss / static_cast<double>(totalSamples));
        metrics.entropy = static_cast<float>(sumEntropy / static_cast<double>(totalSamples));
        metrics.approxKL = static_cast<float>(sumKL / static_cast<double>(totalSamples));
        metrics.clipFraction = static_cast<float>(sumClipped / static_cast<double>(totalSamples));
    }
    if (totalBatches > 0) {
        metrics.gradNorm = static_cast<float>(sumGradNorm / static_cast<double>(totalBatches));
    }
    metrics.explainedVariance = explainedVar;
    metrics.learningRate = lr;
    metrics.entropyCoef = entCoef;
    metrics.updateCount = m_updateCount;
    metrics.klEarlyStopped = earlyStopped;
    m_lastMetrics = metrics;
    return metrics;
}

bool PPOAgent::saveModel(const std::filesystem::path& path) const {
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;

    std::uint64_t checksum = kFnvOffset;
    writePod(os, kModelMagic);
    writePod(os, kModelVersion);
    writePod(os, m_totalSteps);
    writePod(os, m_updateCount);
    checksum = fnv1aUpdate(checksum, &kModelMagic, sizeof(kModelMagic));
    checksum = fnv1aUpdate(checksum, &kModelVersion, sizeof(kModelVersion));
    checksum = fnv1aUpdate(checksum, &m_totalSteps, sizeof(m_totalSteps));
    checksum = fnv1aUpdate(checksum, &m_updateCount, sizeof(m_updateCount));

    m_net.save(os, checksum);
    m_obsNorm.save(os, checksum);
    writePod(os, checksum);
    return static_cast<bool>(os);
}

bool PPOAgent::loadModel(const std::filesystem::path& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint64_t steps = 0;
    std::uint64_t updates = 0;
    if (!readPod(is, magic) || !readPod(is, version) || !readPod(is, steps) || !readPod(is, updates)) {
        return false;
    }
    if (magic != kModelMagic || (version != kModelVersion && version != 1)) {
        return false;
    }

    std::uint64_t checksum = kFnvOffset;
    checksum = fnv1aUpdate(checksum, &magic, sizeof(magic));
    checksum = fnv1aUpdate(checksum, &version, sizeof(version));
    checksum = fnv1aUpdate(checksum, &steps, sizeof(steps));
    checksum = fnv1aUpdate(checksum, &updates, sizeof(updates));

    PolicyNetwork tmpNet(m_config.obsDim, m_config.hidden1, m_config.hidden2, m_config.seed);
    RunningMeanStd tmpNorm(m_config.obsDim, m_config.obsNormDim);
    if (!tmpNet.load(is, checksum) || !tmpNorm.load(is, checksum)) {
        return false;
    }

    std::uint64_t storedChecksum = 0;
    if (!readPod(is, storedChecksum) || storedChecksum != checksum) {
        return false;
    }

    m_net = std::move(tmpNet);
    m_obsNorm = std::move(tmpNorm);
    m_optim.registerParams(m_net.paramSlots());
    m_buffer.clear();
    m_pendingEpisode.clear();
    m_totalSteps = steps;
    m_updateCount = updates;
    return true;
}

} // namespace gdai::rl
