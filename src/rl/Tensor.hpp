#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace gdai::rl {

using Vec = std::vector<float>;

struct Mat {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> data;

    Mat() = default;
    Mat(std::size_t r, std::size_t c, float initVal = 0.0f)
        : rows(r), cols(c), data(r * c, initVal) {}

    float& operator()(std::size_t r, std::size_t c) { return data[r * cols + c]; }
    float operator()(std::size_t r, std::size_t c) const { return data[r * cols + c]; }

    void zero() { std::fill(data.begin(), data.end(), 0.0f); }
    std::size_t size() const { return data.size(); }
};

/// Orthogonal initialization using modified Gram-Schmidt QR decomposition.
/// Improves PPO convergence stability dramatically over naive uniform/Xavier init.
inline void initOrthogonal(Mat& w, float gain, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const std::size_t m = w.rows;
    const std::size_t n = w.cols;
    const bool transpose = m < n;
    const std::size_t r = transpose ? n : m;
    const std::size_t c = transpose ? m : n;

    // Generate r x c matrix with r >= c and orthonormalize its columns
    std::vector<float> q(r * c, 0.0f);
    for (float& v : q) {
        v = dist(rng);
    }

    for (std::size_t j = 0; j < c; ++j) {
        for (std::size_t k = 0; k < j; ++k) {
            double dot = 0.0;
            for (std::size_t i = 0; i < r; ++i) {
                dot += static_cast<double>(q[i * c + k]) * static_cast<double>(q[i * c + j]);
            }
            for (std::size_t i = 0; i < r; ++i) {
                q[i * c + j] -= static_cast<float>(dot) * q[i * c + k];
            }
        }
        double normSq = 0.0;
        for (std::size_t i = 0; i < r; ++i) {
            normSq += static_cast<double>(q[i * c + j]) * static_cast<double>(q[i * c + j]);
        }
        float invNorm = normSq > 1e-12 ? static_cast<float>(1.0 / std::sqrt(normSq)) : 0.0f;
        for (std::size_t i = 0; i < r; ++i) {
            q[i * c + j] *= invNorm;
        }
    }

    if (!transpose) {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                w(i, j) = q[i * c + j] * gain;
            }
        }
    } else {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                w(i, j) = q[j * c + i] * gain;
            }
        }
    }
}

/// Linear forward: y = W * x + b
inline void linearForward(const Mat& w, const Vec& b, const float* x, float* y) {
    const std::size_t outDim = w.rows;
    const std::size_t inDim = w.cols;
    for (std::size_t i = 0; i < outDim; ++i) {
        const float* row = w.data.data() + i * inDim;
        float acc = b[i];
        for (std::size_t j = 0; j < inDim; ++j) {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

/// Linear backward accumulating gradients into dW, db, and writing dx.
inline void linearBackward(
    const Mat& w,
    const float* x,
    const float* dy,
    Mat& dW,
    Vec& db,
    float* dx
) {
    const std::size_t outDim = w.rows;
    const std::size_t inDim = w.cols;
    if (dx) {
        std::fill(dx, dx + inDim, 0.0f);
    }
    for (std::size_t i = 0; i < outDim; ++i) {
        const float g = dy[i];
        db[i] += g;
        float* dRow = dW.data.data() + i * inDim;
        const float* wRow = w.data.data() + i * inDim;
        for (std::size_t j = 0; j < inDim; ++j) {
            dRow[j] += g * x[j];
        }
        if (dx) {
            for (std::size_t j = 0; j < inDim; ++j) {
                dx[j] += wRow[j] * g;
            }
        }
    }
}

/// Numerically stable 2-class softmax -> returns P(action == 1) (HOLD probability)
inline float sigmoidDiff(float logit0, float logit1) {
    const float d = std::clamp(logit1 - logit0, -30.0f, 30.0f);
    return 1.0f / (1.0f + std::exp(-d));
}

inline void writePod(std::ostream& os, const auto& val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

inline bool readPod(std::istream& is, auto& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    return static_cast<bool>(is);
}

inline std::uint64_t fnv1aUpdate(std::uint64_t hash, const void* ptr, std::size_t bytes) {
    const auto* p = static_cast<const std::uint8_t*>(ptr);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<std::uint64_t>(p[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace gdai::rl
