#include "PolicyNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gdai::rl {

PolicyNetwork::PolicyNetwork(std::size_t obsDim, std::size_t hidden1, std::size_t hidden2, std::uint32_t seed)
    : m_obsDim(obsDim)
    , m_hidden1(hidden1)
    , m_hidden2(hidden2)
    , m_aW1(hidden1, obsDim), m_aW2(hidden2, hidden1), m_aW3(3, hidden2)
    , m_aB1(hidden1, 0.0f), m_aB2(hidden2, 0.0f), m_aB3(3, 0.0f)
    , m_daW1(hidden1, obsDim), m_daW2(hidden2, hidden1), m_daW3(3, hidden2)
    , m_daB1(hidden1, 0.0f), m_daB2(hidden2, 0.0f), m_daB3(3, 0.0f)
    , m_cW1(hidden1, obsDim), m_cW2(hidden2, hidden1), m_cW3(1, hidden2)
    , m_cB1(hidden1, 0.0f), m_cB2(hidden2, 0.0f), m_cB3(1, 0.0f)
    , m_dcW1(hidden1, obsDim), m_dcW2(hidden2, hidden1), m_dcW3(1, hidden2)
    , m_dcB1(hidden1, 0.0f), m_dcB2(hidden2, 0.0f), m_dcB3(1, 0.0f)
    , m_scratchH1(hidden1, 0.0f)
    , m_scratchH2(hidden2, 0.0f) {
    resetWeights(seed);
}

void PolicyNetwork::resetWeights(std::uint32_t seed) {
    std::mt19937 rng(seed);
    constexpr float kHiddenGain = 1.41421356237f; // sqrt(2)

    initOrthogonal(m_aW1, kHiddenGain, rng);
    initOrthogonal(m_aW2, kHiddenGain, rng);
    initOrthogonal(m_aW3, 0.01f, rng); // near-zero output projection

    initOrthogonal(m_cW1, kHiddenGain, rng);
    initOrthogonal(m_cW2, kHiddenGain, rng);
    initOrthogonal(m_cW3, 1.0f, rng);

    std::fill(m_aB1.begin(), m_aB1.end(), 0.0f);
    std::fill(m_aB2.begin(), m_aB2.end(), 0.0f);

    // 3-way action biases: Release=0, Click=1, Hold=2
    // User requirement: Click/Release/Hold with significantly reduced hold probability (<5% initially)
    setActionBiases(1.2f, -0.6f, -2.2f);

    std::fill(m_cB1.begin(), m_cB1.end(), 0.0f);
    std::fill(m_cB2.begin(), m_cB2.end(), 0.0f);
    std::fill(m_cB3.begin(), m_cB3.end(), 0.0f);
}

void PolicyNetwork::setActionBiases(float releaseBias, float clickBias, float holdBias) {
    if (m_aB3.size() >= 3) {
        m_aB3[0] = releaseBias;
        m_aB3[1] = clickBias;
        m_aB3[2] = holdBias;
    }
}

ForwardCache PolicyNetwork::forward(const Vec& obs) const {
    ForwardCache cache;
    cache.aH1.resize(m_hidden1);
    cache.aH2.resize(m_hidden2);
    cache.cH1.resize(m_hidden1);
    cache.cH2.resize(m_hidden2);

    // Actor forward
    linearForward(m_aW1, m_aB1, obs.data(), cache.aH1.data());
    for (float& v : cache.aH1) v = std::tanh(v);

    linearForward(m_aW2, m_aB2, cache.aH1.data(), cache.aH2.data());
    for (float& v : cache.aH2) v = std::tanh(v);

    float logits[3] = {0.0f, 0.0f, 0.0f};
    linearForward(m_aW3, m_aB3, cache.aH2.data(), logits);
    cache.logits[0] = logits[0];
    cache.logits[1] = logits[1];
    cache.logits[2] = logits[2];
    cache.logit0 = logits[0];
    cache.logit1 = logits[1];
    cache.logit2 = logits[2];

    // Numerically stable 3-class softmax
    const float maxLogit = std::max({logits[0], logits[1], logits[2]});
    const float exp0 = std::exp(logits[0] - maxLogit);
    const float exp1 = std::exp(logits[1] - maxLogit);
    const float exp2 = std::exp(logits[2] - maxLogit);
    const float sumExp = exp0 + exp1 + exp2;
    const float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : (1.0f / 3.0f);

    cache.probs[0] = exp0 * invSum;
    cache.probs[1] = exp1 * invSum;
    cache.probs[2] = exp2 * invSum;
    cache.releaseProb = cache.probs[0];
    cache.clickProb   = cache.probs[1];
    cache.holdProb    = cache.probs[2];

    // Critic forward
    linearForward(m_cW1, m_cB1, obs.data(), cache.cH1.data());
    for (float& v : cache.cH1) v = std::tanh(v);

    linearForward(m_cW2, m_cB2, cache.cH1.data(), cache.cH2.data());
    for (float& v : cache.cH2) v = std::tanh(v);

    float valOut[1] = {0.0f};
    linearForward(m_cW3, m_cB3, cache.cH2.data(), valOut);
    cache.value = valOut[0];

    return cache;
}

void PolicyNetwork::backwardAccumulate(
    const Vec& obs,
    const ForwardCache& cache,
    float dLogit0,
    float dLogit1,
    float dLogit2,
    float dValue
) {
    // --- Actor backward ---
    if (dLogit0 != 0.0f || dLogit1 != 0.0f || dLogit2 != 0.0f) {
        const float dLogits[3] = {dLogit0, dLogit1, dLogit2};
        linearBackward(m_aW3, cache.aH2.data(), dLogits, m_daW3, m_daB3, m_scratchH2.data());
        for (std::size_t i = 0; i < m_hidden2; ++i) {
            const float h = cache.aH2[i];
            m_scratchH2[i] *= (1.0f - h * h);
        }
        linearBackward(m_aW2, cache.aH1.data(), m_scratchH2.data(), m_daW2, m_daB2, m_scratchH1.data());
        for (std::size_t i = 0; i < m_hidden1; ++i) {
            const float h = cache.aH1[i];
            m_scratchH1[i] *= (1.0f - h * h);
        }
        linearBackward(m_aW1, obs.data(), m_scratchH1.data(), m_daW1, m_daB1, nullptr);
    }

    // --- Critic backward ---
    if (dValue != 0.0f) {
        const float dVal[1] = {dValue};
        linearBackward(m_cW3, cache.cH2.data(), dVal, m_dcW3, m_dcB3, m_scratchH2.data());
        for (std::size_t i = 0; i < m_hidden2; ++i) {
            const float h = cache.cH2[i];
            m_scratchH2[i] *= (1.0f - h * h);
        }
        linearBackward(m_cW2, cache.cH1.data(), m_scratchH2.data(), m_dcW2, m_dcB2, m_scratchH1.data());
        for (std::size_t i = 0; i < m_hidden1; ++i) {
            const float h = cache.cH1[i];
            m_scratchH1[i] *= (1.0f - h * h);
        }
        linearBackward(m_cW1, obs.data(), m_scratchH1.data(), m_dcW1, m_dcB1, nullptr);
    }
}

std::vector<ParamSlot> PolicyNetwork::paramSlots() {
    return {
        {&m_aW1.data, &m_daW1.data, true},
        {&m_aB1,      &m_daB1,      false},
        {&m_aW2.data, &m_daW2.data, true},
        {&m_aB2,      &m_daB2,      false},
        {&m_aW3.data, &m_daW3.data, true},
        {&m_aB3,      &m_daB3,      false},
        {&m_cW1.data, &m_dcW1.data, true},
        {&m_cB1,      &m_dcB1,      false},
        {&m_cW2.data, &m_dcW2.data, true},
        {&m_cB2,      &m_dcB2,      false},
        {&m_cW3.data, &m_dcW3.data, true},
        {&m_cB3,      &m_dcB3,      false},
    };
}

std::size_t PolicyNetwork::parameterCount() const {
    return m_aW1.size() + m_aB1.size() + m_aW2.size() + m_aB2.size() + m_aW3.size() + m_aB3.size()
         + m_cW1.size() + m_cB1.size() + m_cW2.size() + m_cB2.size() + m_cW3.size() + m_cB3.size();
}

void PolicyNetwork::save(std::ostream& os, std::uint64_t& checksum) const {
    auto writeVec = [&](const std::vector<float>& v) {
        const auto bytes = static_cast<std::streamsize>(v.size() * sizeof(float));
        os.write(reinterpret_cast<const char*>(v.data()), bytes);
        checksum = fnv1aUpdate(checksum, v.data(), static_cast<std::size_t>(bytes));
    };

    const std::uint64_t dims[3] = {m_obsDim, m_hidden1, m_hidden2};
    os.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    checksum = fnv1aUpdate(checksum, dims, sizeof(dims));

    writeVec(m_aW1.data); writeVec(m_aB1);
    writeVec(m_aW2.data); writeVec(m_aB2);
    writeVec(m_aW3.data); writeVec(m_aB3);
    writeVec(m_cW1.data); writeVec(m_cB1);
    writeVec(m_cW2.data); writeVec(m_cB2);
    writeVec(m_cW3.data); writeVec(m_cB3);
}

bool PolicyNetwork::load(std::istream& is, std::uint64_t& checksum) {
    std::uint64_t dims[3] = {0, 0, 0};
    is.read(reinterpret_cast<char*>(dims), sizeof(dims));
    if (!is) return false;
    if (dims[0] != m_obsDim || dims[1] != m_hidden1 || dims[2] != m_hidden2) {
        return false;
    }
    checksum = fnv1aUpdate(checksum, dims, sizeof(dims));

    auto readVec = [&](std::vector<float>& v) -> bool {
        const auto bytes = static_cast<std::streamsize>(v.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(v.data()), bytes);
        if (!is) return false;
        checksum = fnv1aUpdate(checksum, v.data(), static_cast<std::size_t>(bytes));
        return true;
    };

    return readVec(m_aW1.data) && readVec(m_aB1)
        && readVec(m_aW2.data) && readVec(m_aB2)
        && readVec(m_aW3.data) && readVec(m_aB3)
        && readVec(m_cW1.data) && readVec(m_cB1)
        && readVec(m_cW2.data) && readVec(m_cB2)
        && readVec(m_cW3.data) && readVec(m_cB3);
}

} // namespace gdai::rl
