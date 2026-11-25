#include "VisionOverlay.hpp"
#include "../game/AIManager.hpp"

using namespace geode::prelude;

namespace gdai::ui {

VisionOverlay* VisionOverlay::create() {
    auto* ret = new VisionOverlay();
    if (ret && ret->init()) {
        ret->autorelease();
        return ret;
    }
    CC_SAFE_DELETE(ret);
    return nullptr;
}

bool VisionOverlay::init() {
    if (!CCNode::init()) return false;
    m_draw = CCDrawNode::create();
    this->addChild(m_draw);
    return true;
}

void VisionOverlay::refresh(PlayLayer* playLayer) {
    if (!m_draw) return;
    m_draw->clear();

    const auto& mgr = game::AIManager::get();
    if (!mgr.showVision() || !mgr.isAIEnabled() || !mgr.hasValidObservation() || !playLayer || !playLayer->m_player1) {
        return;
    }

    const auto& obs = mgr.latestObservation();
    const auto& grid = obs.grid;
    const float cell = env::VisionGrid::kCellSize;
    const float totalW = static_cast<float>(env::VisionGrid::kCols) * cell;
    const float totalH = static_cast<float>(env::VisionGrid::kRows) * cell;

    // Outer vision bounds
    CCPoint bounds[4] = {
        {grid.originX, grid.originY},
        {grid.originX + totalW, grid.originY},
        {grid.originX + totalW, grid.originY + totalH},
        {grid.originX, grid.originY + totalH}
    };
    m_draw->drawPolygon(
        bounds, 4,
        ccc4f(0.05f, 0.12f, 0.20f, 0.06f),
        0.6f,
        ccc4f(0.35f, 0.85f, 1.0f, 0.30f)
    );

    // Draw active perception cells
    for (std::size_t r = 0; r < env::VisionGrid::kRows; ++r) {
        for (std::size_t c = 0; c < env::VisionGrid::kCols; ++c) {
            const bool isHazard = grid.at(c, r, 0) > 0.5f;
            const bool isSolid = grid.at(c, r, 1) > 0.5f;
            const bool isInter = grid.at(c, r, 2) > 0.5f;
            if (!isHazard && !isSolid && !isInter) continue;

            const float x0 = grid.originX + static_cast<float>(c) * cell + 1.5f;
            const float y0 = grid.originY + static_cast<float>(r) * cell + 1.5f;
            const float x1 = x0 + cell - 3.0f;
            const float y1 = y0 + cell - 3.0f;

            CCPoint quad[4] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};

            ccColor4F fill{0, 0, 0, 0};
            ccColor4F border{0, 0, 0, 0};
            if (isHazard) {
                fill = ccc4f(1.0f, 0.18f, 0.22f, 0.34f);
                border = ccc4f(1.0f, 0.35f, 0.35f, 0.70f);
            } else if (isInter) {
                fill = ccc4f(1.0f, 0.82f, 0.15f, 0.36f);
                border = ccc4f(1.0f, 0.90f, 0.30f, 0.75f);
            } else if (isSolid) {
                fill = ccc4f(0.20f, 0.85f, 0.58f, 0.20f);
                border = ccc4f(0.25f, 0.95f, 0.65f, 0.45f);
            }
            m_draw->drawPolygon(quad, 4, fill, 0.6f, border);
        }
    }

    // Draw precision vector rays from player center to nearest hazard & interactive target
    const CCPoint pPos = playLayer->m_player1->getPosition();
    if (obs.targets.hasHazard1) {
        const CCPoint hPos{pPos.x + obs.targets.hazard1Dx, pPos.y + obs.targets.hazard1Dy};
        m_draw->drawSegment(pPos, hPos, 0.9f, ccc4f(1.0f, 0.22f, 0.22f, 0.72f));
        m_draw->drawDot(hPos, 2.5f, ccc4f(1.0f, 0.25f, 0.25f, 0.90f));
    }
    if (obs.targets.hasInteractive) {
        const CCPoint iPos{pPos.x + obs.targets.interactiveDx, pPos.y + obs.targets.interactiveDy};
        m_draw->drawSegment(pPos, iPos, 0.8f, ccc4f(1.0f, 0.84f, 0.18f, 0.70f));
        m_draw->drawDot(iPos, 2.2f, ccc4f(1.0f, 0.88f, 0.25f, 0.90f));
    }
}

} // namespace gdai::ui
