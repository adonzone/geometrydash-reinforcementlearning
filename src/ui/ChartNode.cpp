#include "ChartNode.hpp"

#include <algorithm>
#include <cmath>

using namespace geode::prelude;

namespace gdai::ui {

ChartNode* ChartNode::create(float width, float height) {
    auto* ret = new ChartNode();
    if (ret && ret->init(width, height)) {
        ret->autorelease();
        return ret;
    }
    CC_SAFE_DELETE(ret);
    return nullptr;
}

bool ChartNode::init(float width, float height) {
    if (!CCNode::init()) return false;
    m_width = width;
    m_height = height;
    this->setContentSize({width, height});

    m_draw = CCDrawNode::create();
    this->addChild(m_draw);
    return true;
}

void ChartNode::setFixedRange(float minY, float maxY) {
    m_fixedRange = true;
    m_minY = minY;
    m_maxY = maxY;
}

void ChartNode::setAutoRange() {
    m_fixedRange = false;
}

void ChartNode::renderSeries(const std::vector<ChartSeries>& seriesList) {
    if (!m_draw) return;
    m_draw->clear();

    // Dark translucent chart frame
    CCPoint bg[4] = {
        {0.f, 0.f},
        {m_width, 0.f},
        {m_width, m_height},
        {0.f, m_height}
    };
    m_draw->drawPolygon(
        bg, 4,
        ccc4f(0.05f, 0.07f, 0.11f, 0.78f),
        0.7f,
        ccc4f(0.32f, 0.42f, 0.56f, 0.55f)
    );

    // Subtle horizontal reference grid lines (25%, 50%, 75%)
    for (int i = 1; i <= 3; ++i) {
        const float y = m_height * (static_cast<float>(i) * 0.25f);
        m_draw->drawSegment({1.f, y}, {m_width - 1.f, y}, 0.4f, ccc4f(1.f, 1.f, 1.f, 0.10f));
    }

    float minV = m_minY;
    float maxV = m_maxY;
    if (!m_fixedRange) {
        minV = 1e9f;
        maxV = -1e9f;
        for (const auto& s : seriesList) {
            for (float v : s.values) {
                minV = std::min(minV, v);
                maxV = std::max(maxV, v);
            }
        }
        if (minV > maxV) {
            minV = 0.0f;
            maxV = 1.0f;
        } else if (std::abs(maxV - minV) < 1e-4f) {
            minV -= 1.0f;
            maxV += 1.0f;
        }
    }

    const float pad = 3.0f;
    const float usableW = std::max(1.0f, m_width - 2.0f * pad);
    const float usableH = std::max(1.0f, m_height - 2.0f * pad);
    const float span = std::max(1e-5f, maxV - minV);

    for (const auto& s : seriesList) {
        if (s.values.size() < 2) {
            if (s.values.size() == 1) {
                const float norm = std::clamp((s.values[0] - minV) / span, 0.0f, 1.0f);
                m_draw->drawDot({pad + usableW * 0.5f, pad + norm * usableH}, 2.0f, s.color);
            }
            continue;
        }

        const float stepX = usableW / static_cast<float>(s.values.size() - 1);
        CCPoint prev{};
        for (std::size_t i = 0; i < s.values.size(); ++i) {
            const float norm = std::clamp((s.values[i] - minV) / span, 0.0f, 1.0f);
            CCPoint cur{pad + static_cast<float>(i) * stepX, pad + norm * usableH};
            if (i > 0) {
                m_draw->drawSegment(prev, cur, s.lineThickness, s.color);
            }
            prev = cur;
        }
        m_draw->drawDot(prev, 2.2f, s.color);
    }
}

void ChartNode::renderGauge(float fraction, ccColor4F fillColor, bool showMidTick) {
    if (!m_draw) return;
    m_draw->clear();

    const float clamped = std::clamp(fraction, 0.0f, 1.0f);
    CCPoint bg[4] = {
        {0.f, 0.f}, {m_width, 0.f}, {m_width, m_height}, {0.f, m_height}
    };
    m_draw->drawPolygon(
        bg, 4,
        ccc4f(0.08f, 0.10f, 0.14f, 0.85f),
        0.6f,
        ccc4f(0.35f, 0.45f, 0.58f, 0.55f)
    );

    const float fillW = m_width * clamped;
    if (fillW > 1.0f) {
        CCPoint fill[4] = {
            {0.5f, 0.5f},
            {fillW - 0.5f, 0.5f},
            {fillW - 0.5f, m_height - 0.5f},
            {0.5f, m_height - 0.5f}
        };
        m_draw->drawPolygon(fill, 4, fillColor, 0.0f, ccc4f(0, 0, 0, 0));
    }

    if (showMidTick) {
        const float midX = m_width * 0.5f;
        m_draw->drawSegment({midX, 0.f}, {midX, m_height}, 0.6f, ccc4f(1.f, 1.f, 1.f, 0.65f));
    }
}

} // namespace gdai::ui
