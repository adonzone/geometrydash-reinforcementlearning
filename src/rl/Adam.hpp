#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gdai::rl {

struct ParamSlot {
    std::vector<float>* param = nullptr;
    std::vector<float>* grad = nullptr;
    bool applyWeightDecay = true;
};

/// AdamW optimizer with global L2 gradient clipping and bias correction.
class AdamWOptimizer {
public:
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-5f;
    float weightDecay = 1e-4f;

    void registerParams(std::vector<ParamSlot> slots) {
        m_slots = std::move(slots);
        m_m.clear();
        m_v.clear();
        m_m.reserve(m_slots.size());
        m_v.reserve(m_slots.size());
        for (const auto& s : m_slots) {
            m_m.emplace_back(s.param->size(), 0.0f);
            m_v.emplace_back(s.param->size(), 0.0f);
        }
        m_step = 0;
    }

    void resetState() {
        for (auto& m : m_m) std::fill(m.begin(), m.end(), 0.0f);
        for (auto& v : m_v) std::fill(v.begin(), v.end(), 0.0f);
        m_step = 0;
    }

    void zeroGrad() {
        for (auto& s : m_slots) {
            std::fill(s.grad->begin(), s.grad->end(), 0.0f);
        }
    }

    /// Clips global gradient norm to maxGradNorm and applies one AdamW step.
    /// Returns the pre-clip global L2 gradient norm for diagnostics.
    float step(float lr, float maxGradNorm) {
        double normSq = 0.0;
        for (const auto& s : m_slots) {
            for (float g : *s.grad) {
                normSq += static_cast<double>(g) * static_cast<double>(g);
            }
        }
        const float totalNorm = static_cast<float>(std::sqrt(normSq));
        const float clipScale = (maxGradNorm > 0.0f && totalNorm > maxGradNorm)
            ? (maxGradNorm / (totalNorm + 1e-6f))
            : 1.0f;

        ++m_step;
        const float biasCorr1 = 1.0f - std::pow(beta1, static_cast<float>(m_step));
        const float biasCorr2 = 1.0f - std::pow(beta2, static_cast<float>(m_step));

        for (std::size_t idx = 0; idx < m_slots.size(); ++idx) {
            auto& p = *m_slots[idx].param;
            const auto& g = *m_slots[idx].grad;
            auto& m = m_m[idx];
            auto& v = m_v[idx];
            const bool decay = m_slots[idx].applyWeightDecay;

            for (std::size_t i = 0; i < p.size(); ++i) {
                const float gi = g[i] * clipScale;
                m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
                v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;

                const float mHat = m[i] / biasCorr1;
                const float vHat = v[i] / biasCorr2;
                if (decay && weightDecay > 0.0f) {
                    p[i] -= lr * weightDecay * p[i];
                }
                p[i] -= lr * mHat / (std::sqrt(vHat) + eps);
            }
        }
        return totalNorm;
    }

    std::uint64_t stepCount() const { return m_step; }

private:
    std::vector<ParamSlot> m_slots;
    std::vector<std::vector<float>> m_m;
    std::vector<std::vector<float>> m_v;
    std::uint64_t m_step = 0;
};

} // namespace gdai::rl
