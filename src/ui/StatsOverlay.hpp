#pragma once

#include "ChartNode.hpp"

#include <Geode/Geode.hpp>

namespace gdai::ui {

/// Compact screen-space HUD overlay inside PlayLayer showing real-time RL training metrics and chart.
class StatsOverlay : public cocos2d::CCNode {
public:
    static StatsOverlay* create();
    bool init() override;

    void refresh();

private:
    cocos2d::CCLayerColor* m_bg = nullptr;
    cocos2d::CCLabelBMFont* m_headerLabel = nullptr;
    cocos2d::CCLabelBMFont* m_statusBadge = nullptr;
    cocos2d::CCLabelBMFont* m_line1 = nullptr;
    cocos2d::CCLabelBMFont* m_line2 = nullptr;
    cocos2d::CCLabelBMFont* m_line3 = nullptr;
    cocos2d::CCLabelBMFont* m_line4 = nullptr;
    cocos2d::CCLabelBMFont* m_actionLabel = nullptr;

    ChartNode* m_holdGauge = nullptr;
    ChartNode* m_progressChart = nullptr;
    int m_frameSkip = 0;
};

} // namespace gdai::ui
