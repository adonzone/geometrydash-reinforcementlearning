#pragma once

#include <Geode/Geode.hpp>

namespace gdai::ui {

/// World-space overlay inside PlayLayer->m_objectLayer rendering the AI's live perception grid.
class VisionOverlay : public cocos2d::CCNode {
public:
    static VisionOverlay* create();
    bool init() override;

    void refresh(PlayLayer* playLayer);

private:
    cocos2d::CCDrawNode* m_draw = nullptr;
};

} // namespace gdai::ui
