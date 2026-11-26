#pragma once

#include "Tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gdai::rl {

struct Transition {
    Vec normObs;
    int action = 0;           // 0 = RELEASE, 1 = HOLD
    float logProb = 0.0f;
    float value = 0.0f;
    float reward = 0.0f;
    bool done = false;        // true on real terminal (death or level complete)
    bool truncated = false;   // true on gamemode hot-swap segment boundary (bootstrap from next mode)
    float bootstrapValue = 0.0f;
    std::uint64_t episodeId = 0;
};

/// On-policy trajectory buffer with segment-aware Generalized Advantage Estimation (GAE-lambda).
class RolloutBuffer {
public:
    explicit RolloutBuffer(std::size_t capacity = 512) : m_capacity(capacity) {
        m_items.reserve(capacity);
        m_advantages.reserve(capacity);
        m_returns.reserve(capacity);
    }

    void resize(std::size_t capacity) {
        m_capacity = capacity;
        clear();
        m_items.reserve(capacity);
        m_advantages.reserve(capacity);
        m_returns.reserve(capacity);
    }

    void add(Transition t) {
        m_items.push_back(std::move(t));
    }

    bool full() const { return m_items.size() >= m_capacity; }
    std::size_t size() const { return m_items.size(); }
    std::size_t capacity() const { return m_capacity; }
    bool empty() const { return m_items.empty(); }

    void clear() {
        m_items.clear();
        m_advantages.clear();
        m_returns.clear();
    }

    /// Computes GAE advantages and lambda-returns backwards from bootstrapValue.
    /// Respects both true terminal boundaries (done) and gamemode hot-swap boundaries (truncated).
    void computeGAE(float bootstrapValue, float gamma, float gaeLambda) {
        const std::size_t n = m_items.size();
        m_advantages.assign(n, 0.0f);
        m_returns.assign(n, 0.0f);
        if (n == 0) return;

        float lastGae = 0.0f;
        for (std::size_t rev = 0; rev < n; ++rev) {
            const std::size_t t = n - 1 - rev;
            const auto& tr = m_items[t];

            if (tr.done) {
                const float delta = tr.reward - tr.value;
                lastGae = delta;
            } else if (tr.truncated) {
                // Gamemode portal swap: bootstrap from the next gamemode critic, but do not
                // carry GAE across the discontinuity to unrelated future segments of this mode.
                const float delta = tr.reward + gamma * tr.bootstrapValue - tr.value;
                lastGae = delta;
            } else {
                const float nextValue = (t + 1 == n) ? bootstrapValue : m_items[t + 1].value;
                const float delta = tr.reward + gamma * nextValue - tr.value;
                lastGae = delta + gamma * gaeLambda * lastGae;
            }

            m_advantages[t] = lastGae;
            m_returns[t] = lastGae + tr.value;
        }
    }

    const Transition& item(std::size_t i) const { return m_items[i]; }
    float advantage(std::size_t i) const { return m_advantages[i]; }
    float ret(std::size_t i) const { return m_returns[i]; }

private:
    std::size_t m_capacity;
    std::vector<Transition> m_items;
    std::vector<float> m_advantages;
    std::vector<float> m_returns;
};

} // namespace gdai::rl
