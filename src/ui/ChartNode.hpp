#pragma once

#include <Geode/Geode.hpp>
#include <vector>

namespace gdai::ui {

struct ChartSeries {
    std::vector<float> values;
    cocos2d::ccColor4F color{0.25f, 0.95f, 0.55f, 1.0f};
    float lineThickness = 1.2f;
};

/// Lightweight Cocos2d CCDrawNode-based multi-series chart & bar gauge.
class ChartNode : public cocos2d::CCNode {
public:
    static ChartNode* create(float width, float height);

    bool init(float width, float height);

    void setFixedRange(float minY, float maxY);
    void setAutoRange();

    /// Redraws the chart with one or more time-series lines.
    void renderSeries(const std::vector<ChartSeries>& seriesList);

    /// Draws a horizontal filled gauge bar [0..1] with a 0.5 midpoint marker.
    void renderGauge(float fraction, cocos2d::ccColor4F fillColor, bool showMidTick = true);

private:
    float m_width = 160.0f;
    float m_height = 50.0f;
    bool m_fixedRange = false;
    float m_minY = 0.0f;
    float m_maxY = 100.0f;
    cocos2d::CCDrawNode* m_draw = nullptr;
};

} // namespace gdai::ui
