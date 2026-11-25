#pragma once

#include "Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

namespace gdai::rl {

/// Welford online per-feature observation normalizer.
class RunningMeanStd {
public:
    explicit RunningMeanStd(std::size_t dim = 0) { reset(dim); }

    void reset(std::size_t dim) {
        m_dim = dim;
        m_count = 0;
        m_mean.assign(dim, 0.0);
        m_m2.assign(dim, 1.0);
    }

    void update(const Vec& x) {
        if (x.size() != m_dim) return;
        ++m_count;
        const double invN = 1.0 / static_cast<double>(m_count);
        for (std::size_t i = 0; i < m_dim; ++i) {
            const double val = static_cast<double>(x[i]);
            const double delta = val - m_mean[i];
            m_mean[i] += delta * invN;
            const double delta2 = val - m_mean[i];
            m_m2[i] += delta * delta2;
        }
    }

    Vec normalize(const Vec& x, float clip = 10.0f) const {
        Vec out(m_dim, 0.0f);
        if (x.size() != m_dim) return out;
        if (m_count < 2) {
            for (std::size_t i = 0; i < m_dim; ++i) {
                out[i] = std::clamp(x[i], -clip, clip);
            }
            return out;
        }
        const double denom = static_cast<double>(m_count - 1);
        for (std::size_t i = 0; i < m_dim; ++i) {
            const double var = std::max(m_m2[i] / denom, 1e-4);
            const double norm = (static_cast<double>(x[i]) - m_mean[i]) / std::sqrt(var);
            out[i] = std::clamp(static_cast<float>(norm), -clip, clip);
        }
        return out;
    }

    std::size_t dim() const { return m_dim; }
    std::uint64_t count() const { return m_count; }

    void save(std::ostream& os, std::uint64_t& checksum) const {
        std::uint64_t d = m_dim;
        writePod(os, d);
        writePod(os, m_count);
        checksum = fnv1aUpdate(checksum, &d, sizeof(d));
        checksum = fnv1aUpdate(checksum, &m_count, sizeof(m_count));
        if (m_dim > 0) {
            os.write(reinterpret_cast<const char*>(m_mean.data()), static_cast<std::streamsize>(m_dim * sizeof(double)));
            os.write(reinterpret_cast<const char*>(m_m2.data()), static_cast<std::streamsize>(m_dim * sizeof(double)));
            checksum = fnv1aUpdate(checksum, m_mean.data(), m_dim * sizeof(double));
            checksum = fnv1aUpdate(checksum, m_m2.data(), m_dim * sizeof(double));
        }
    }

    bool load(std::istream& is, std::uint64_t& checksum) {
        std::uint64_t d = 0;
        std::uint64_t c = 0;
        if (!readPod(is, d) || !readPod(is, c)) return false;
        if (d != m_dim) return false;
        checksum = fnv1aUpdate(checksum, &d, sizeof(d));
        checksum = fnv1aUpdate(checksum, &c, sizeof(c));
        m_count = c;
        if (m_dim > 0) {
            is.read(reinterpret_cast<char*>(m_mean.data()), static_cast<std::streamsize>(m_dim * sizeof(double)));
            is.read(reinterpret_cast<char*>(m_m2.data()), static_cast<std::streamsize>(m_dim * sizeof(double)));
            if (!is) return false;
            checksum = fnv1aUpdate(checksum, m_mean.data(), m_dim * sizeof(double));
            checksum = fnv1aUpdate(checksum, m_m2.data(), m_dim * sizeof(double));
        }
        return true;
    }

private:
    std::size_t m_dim = 0;
    std::uint64_t m_count = 0;
    std::vector<double> m_mean;
    std::vector<double> m_m2;
};

} // namespace gdai::rl
