#pragma once

#include "ChartNode.hpp"

#include <Geode/Geode.hpp>

namespace gdai::ui {

/// Full Geode v6 Popup Dashboard for controlling training, viewing charts/stats, and managing checkpoints.
class AIDashboardPopup : public geode::Popup {
public:
    static AIDashboardPopup* create();

protected:
    bool init() override;

private:
    enum class Tab { Controls = 0, Analytics = 1, Models = 2 };
    enum class ChartMetric { Progress = 0, Reward = 1, PPODiagnostics = 2 };

    void switchTab(Tab tab);
    void rebuildTabContent();
    void buildControlsTab(cocos2d::CCNode* parent);
    void buildAnalyticsTab(cocos2d::CCNode* parent);
    void buildModelsTab(cocos2d::CCNode* parent);

    Tab m_activeTab = Tab::Controls;
    ChartMetric m_chartMetric = ChartMetric::Progress;
    cocos2d::CCNode* m_contentLayer = nullptr;
    cocos2d::CCLabelBMFont* m_toastLabel = nullptr;
};

} // namespace gdai::ui
