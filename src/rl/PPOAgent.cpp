#include "PPOAgent.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace gdai::rl {

namespace {
constexpr std::uint32_t kModelMagic = 0x49414447; // "GDAI"
constexpr std::uint32_t kModelVersion = 1;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
} // namespace

PPOConfig PPOConfig::fromPreset(TrainingPreset preset, std::size_t obsDim) {
    PPOConfig cfg;
    cfg.obsDim = obsDim;
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
            break;
    }
    return cfg;
}

PPOAgent::PPOAgent(PPOConfig config)
    : m_config(config)
    , m_net(config.obsDim, config.hidden1, config.hidden2, config.seed)
    , m_obsNorm(config.obsDim)
    , m_buffer(config.rolloutHorizon)
    , m_rng(config.seed + 9973u) {
    m_optim.registerParams(m_net.paramSlots());
}

ActionDecision PPOAgent::selectAction(const Vec& rawObs, bool training) {
    if (training && m_config.normalizeObs) {
        m_obsNorm.update(rawObs);
    }
    Vec normObs = m_config.normalizeObs ? m_obsNorm.normalize(rawObs) : rawObs;
    const ForwardCache cache = m_net.forward(normObs);

    const float p1 = std::clamp(cache.holdProb, 1e-6f, 1.0f - 1e-6f);
    const float p0 = 1.0f - p1;

    int action = 0;
    if (training) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        action = (u(m_rng) < p1) ? 1 : 0;
    } else {
        action = (p1 >= 0.5f) ? 1 : 0;
    }

    const float logProb = std::log(action == 1 ? p1 : p0);

    ActionDecision out;
    out.action = action;
    out.holdProb = p1;
    out.logProb = logProb;
    out.value = cache.value;
    out.normObs = std::move(normObs);
    return out;
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

std::optional<UpdateMetrics> PPOAgent::forceUpdate(float bootstrapValue) {
    if (m_buffer.size() < std::max<std::size_t>(16, m_config.minibatchSize / 2)) {
        return std::nullopt;
    }
    return runPPOUpdate(bootstrapValue);
}

void PPOAgent::clearBuffer() {
    m_buffer.clear();
}

void PPOAgent::reset(std::uint32_t newSeed) {
    if (newSeed != 0) {
        m_config.seed = newSeed;
    }
    m_net.resetWeights(m_config.seed);
    m_optim.resetState();
    m_obsNorm.reset(m_config.obsDim);
    m_buffer.clear();
    m_rng.seed(m_config.seed + 9973u);
    m_totalSteps = 0;
    m_updateCount = 0;
    m_lastMetrics = UpdateMetrics{};
}

void PPOAgent::applyPreset(TrainingPreset preset) {
    const auto p = PPOConfig::fromPreset(preset, m_config.obsDim);
    m_config.rolloutHorizon = p.rolloutHorizon;
    m_config.minibatchSize = p.minibatchSize;
    m_config.epochs = p.epochs;
    m_config.learningRate = p.learningRate;
    m_config.minLearningRate = p.minLearningRate;
    m_config.entropyCoefStart = p.entropyCoefStart;
    m_config.entropyCoefEnd = p.entropyCoefEnd;
    m_config.clipEpsilon = p.clipEpsilon;
    m_config.targetKL = p.targetKL;
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
    // Cosine annealing from learningRate down to minLearningRate
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
                const float p1 = std::clamp(cache.holdProb, 1e-6f, 1.0f - 1e-6f);
                const float p0 = 1.0f - p1;

                const float newLogProb = std::log(tr.action == 1 ? p1 : p0);
                const float logRatio = std::clamp(newLogProb - tr.logProb, -20.0f, 20.0f);
                const float ratio = std::exp(logRatio);

                // Schulman unbiased KL estimator: (ratio - 1) - logRatio
                const float approxKL = (ratio - 1.0f) - logRatio;
                batchKL += approxKL;

                // Clipped PPO surrogate objective (we minimize L = -min(surr1, surr2))
                const float surr1 = ratio * adv;
                const float clippedRatio = std::clamp(ratio, 1.0f - clipEps, 1.0f + clipEps);
                const float surr2 = clippedRatio * adv;
                const bool useUnclipped = (surr1 <= surr2);
                const float policyLoss = -std::min(surr1, surr2);
                if (std::abs(ratio - 1.0f) > clipEps) {
                    sumClipped += 1.0;
                }

                // Gradient of policy loss w.r.t. newLogProb:
                // d(-ratio * adv)/d(newLogProb) = -ratio * adv if active, else 0.
                const float dL_dLogProb = useUnclipped ? (-ratio * adv) : 0.0f;

                // Entropy H = -(p0 * log p0 + p1 * log p1)
                const float logP0 = std::log(p0);
                const float logP1 = std::log(p1);
                const float entropy = -(p0 * logP0 + p1 * logP1);

                // Gradient of (-entCoef * H) w.r.t. logits:
                // dH/d(logit1) = p1 * p0 * (logP0 - logP1), dH/d(logit0) = -dH/d(logit1)
                // So d(-entCoef * H)/d(logit1) = -entCoef * p1 * p0 * (logP0 - logP1)
                const float dEnt_dLogit1 = -entCoef * p1 * p0 * (logP0 - logP1);

                // Gradient of newLogProb w.r.t. logits:
                // d(log p_a)/d(logit_i) = (1_{i==a} - p_i)
                const float dLogProb_dLogit0 = (tr.action == 0 ? 1.0f : 0.0f) - p0;
                const float dLogProb_dLogit1 = (tr.action == 1 ? 1.0f : 0.0f) - p1;

                const float dLogit0 = (dL_dLogProb * dLogProb_dLogit0 - dEnt_dLogit1) * invBatch;
                const float dLogit1 = (dL_dLogProb * dLogProb_dLogit1 + dEnt_dLogit1) * invBatch;

                // Clipped value loss: 0.5 * max((V - R)^2, (V_clipped - R)^2)
                const float vDiff = cache.value - targetRet;
                const float vLossUnclipped = 0.5f * vDiff * vDiff;
                const float vClipped = tr.value + std::clamp(cache.value - tr.value, -vClipEps, vClipEps);
                const float vDiffClipped = vClipped - targetRet;
                const float vLossClipped = 0.5f * vDiffClipped * vDiffClipped;

                float vLoss = 0.0f;
                float dValue = 0.0f;
                if (vLossUnclipped >= vLossClipped) {
                    vLoss = vLossUnclipped;
                    dValue = vCoef * vDiff * invBatch;
                } else {
                    vLoss = vLossClipped;
                    // Derivative is non-zero only if value wasn't saturated at the clip boundary
                    const bool withinClip = std::abs(cache.value - tr.value) <= vClipEps;
                    dValue = withinClip ? (vCoef * vDiffClipped * invBatch) : 0.0f;
                }

                m_net.backwardAccumulate(tr.normObs, cache, dLogit0, dLogit1, dValue);

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
    if (magic != kModelMagic || version != kModelVersion) {
        return false;
    }

    std::uint64_t checksum = kFnvOffset;
    checksum = fnv1aUpdate(checksum, &magic, sizeof(magic));
    checksum = fnv1aUpdate(checksum, &version, sizeof(version));
    checksum = fnv1aUpdate(checksum, &steps, sizeof(steps));
    checksum = fnv1aUpdate(checksum, &updates, sizeof(updates));

    PolicyNetwork tmpNet(m_config.obsDim, m_config.hidden1, m_config.hidden2, m_config.seed);
    RunningMeanStd tmpNorm(m_config.obsDim);
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
    m_totalSteps = steps;
    m_updateCount = updates;
    return true;
}

} // namespace gdai::rl
