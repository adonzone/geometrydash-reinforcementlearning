#pragma once

#include "Adam.hpp"
#include "Tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <random>
#include <vector>

namespace gdai::rl {

struct ForwardCache {
    Vec aH1;
    Vec aH2;
    Vec cH1;
    Vec cH2;
    float logit0 = 0.0f;
    float logit1 = 0.0f;
    float holdProb = 0.5f;
    float value = 0.0f;
};

/// Decoupled Actor-Critic MLP with orthogonal initialization and Tanh activations.
/// Actor:  obsDim -> hidden1 -> hidden2 -> 2 logits (RELEASE=0, HOLD=1)
/// Critic: obsDim -> hidden1 -> hidden2 -> 1 scalar value V(s)
class PolicyNetwork {
public:
    PolicyNetwork(std::size_t obsDim, std::size_t hidden1 = 128, std::size_t hidden2 = 64, std::uint32_t seed = 42);

    void resetWeights(std::uint32_t seed);

    /// Fast forward pass with activation cache (for training & inference).
    ForwardCache forward(const Vec& obs) const;

    /// Accumulates gradients for a single transition given scalar sensitivities:
    /// dLogit0 = dL/d(logit0), dLogit1 = dL/d(logit1), dValue = dL/d(value).
    void backwardAccumulate(
        const Vec& obs,
        const ForwardCache& cache,
        float dLogit0,
        float dLogit1,
        float dValue
    );

    std::vector<ParamSlot> paramSlots();

    std::size_t obsDim() const { return m_obsDim; }
    std::size_t hidden1() const { return m_hidden1; }
    std::size_t hidden2() const { return m_hidden2; }
    std::size_t parameterCount() const;

    void save(std::ostream& os, std::uint64_t& checksum) const;
    bool load(std::istream& is, std::uint64_t& checksum);

private:
    std::size_t m_obsDim;
    std::size_t m_hidden1;
    std::size_t m_hidden2;

    // Actor weights & biases
    Mat m_aW1, m_aW2, m_aW3;
    Vec m_aB1, m_aB2, m_aB3;
    Mat m_daW1, m_daW2, m_daW3;
    Vec m_daB1, m_daB2, m_daB3;

    // Critic weights & biases
    Mat m_cW1, m_cW2, m_cW3;
    Vec m_cB1, m_cB2, m_cB3;
    Mat m_dcW1, m_dcW2, m_dcW3;
    Vec m_dcB1, m_dcB2, m_dcB3;

    // Scratch buffers for backward pass (reused to avoid heap churn)
    mutable Vec m_scratchH1;
    mutable Vec m_scratchH2;
};

} // namespace gdai::rl
