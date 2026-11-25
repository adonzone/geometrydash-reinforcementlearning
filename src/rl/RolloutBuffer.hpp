#pragma once

#include "Tensor.hpp"

#include <cstddef>
#include <vector>

namespace gdai::rl {

struct Transition {
    Vec normObs;
    int action = 0;       // 0 = RELEASE, 1 = HOLD
    float logProb = 0.0f;
    float value = 0.0f;
    float reward = 0.0f;
    bool done = false;
};

/// On-policy trajectory buffer with Generalized Advantage Estimation (GAE-lambda).
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
    void computeGAE(float bootstrapValue, float gamma, float gaeLambda) {
        const std::size_t n = m_items.size();
        m_advantages.assign(n, 0.0f);
        m_returns.assign(n, 0.0f);
        if (n == 0) return;

        float lastGae = 0.0f;
        for (std::size_t rev = 0; rev < n; ++rev) {
            const std::size_t t = n - 1 - rev;
            const float nextNonTerminal = m_items[t].done ? 0.0f : 1.0f;
            const float nextValue = (t + 1 == n) ? bootstrapValue : m_items[t + 1].value;
            const float delta = m_items[t].reward + gamma * nextValue * nextNonTerminal - m_items[t].value;
            lastGae = delta + gamma * gaeLambda * nextNonTerminal * lastGae;
            m_advantages[t] = lastGae;
            m_returns[t] = lastGae + m_items[t].value;
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
