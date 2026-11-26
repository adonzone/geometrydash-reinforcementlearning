#include "GameBridge.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

using namespace geode::prelude;

namespace gdai::game {

env::PlayerState GameBridge::extractPlayerState(
    PlayLayer* playLayer,
    bool aiHolding,
    float holdDurationSec
) {
    env::PlayerState s;
    if (!playLayer || !playLayer->m_player1) return s;

    PlayerObject* p = playLayer->m_player1;
    s.x = p->getPositionX();
    s.y = p->getPositionY();
    s.yVelocity = static_cast<float>(p->m_yVelocity);
    s.playerSpeed = p->m_playerSpeed;
    s.rotationDeg = p->getRotation();
    s.isOnGround = p->m_isOnGround;
    s.isUpsideDown = p->m_isUpsideDown;
    s.isMini = p->m_vehicleSize < 0.8f;
    s.isDual = playLayer->m_gameState.m_isDualMode;
    s.isHolding = aiHolding;
    s.holdDurationSec = holdDurationSec;
    s.touchingOrb = (p->m_touchingRings != nullptr && p->m_touchingRings->count() > 0);

    if (p->m_isShip)        s.mode = env::Gamemode::Ship;
    else if (p->m_isBall)   s.mode = env::Gamemode::Ball;
    else if (p->m_isBird)   s.mode = env::Gamemode::UFO;
    else if (p->m_isDart)   s.mode = env::Gamemode::Wave;
    else if (p->m_isRobot)  s.mode = env::Gamemode::Robot;
    else if (p->m_isSpider) s.mode = env::Gamemode::Spider;
    else if (p->m_isSwing)  s.mode = env::Gamemode::Swing;
    else                    s.mode = env::Gamemode::Cube;

    return s;
}

env::ObjectCategory GameBridge::classifyObject(GameObject* obj) {
    if (!obj || obj->m_isHide || obj->m_isDecoration || obj->m_isDecoration2) {
        return env::ObjectCategory::None;
    }

    switch (obj->m_objectType) {
        case GameObjectType::Hazard:
        case GameObjectType::AnimatedHazard:
            return env::ObjectCategory::Hazard;

        case GameObjectType::Solid:
        case GameObjectType::Slope:
        case GameObjectType::Breakable:
            return env::ObjectCategory::Solid;

        case GameObjectType::YellowJumpRing:
        case GameObjectType::PinkJumpRing:
        case GameObjectType::RedJumpRing:
        case GameObjectType::GravityRing:
        case GameObjectType::GreenRing:
        case GameObjectType::DropRing:
        case GameObjectType::CustomRing:
        case GameObjectType::DashRing:
        case GameObjectType::GravityDashRing:
        case GameObjectType::SpiderOrb:
        case GameObjectType::TeleportOrb:
            return env::ObjectCategory::Orb;

        case GameObjectType::YellowJumpPad:
        case GameObjectType::PinkJumpPad:
        case GameObjectType::RedJumpPad:
        case GameObjectType::GravityPad:
        case GameObjectType::SpiderPad:
            return env::ObjectCategory::Pad;

        case GameObjectType::InverseGravityPortal:
        case GameObjectType::NormalGravityPortal:
        case GameObjectType::GravityTogglePortal:
        case GameObjectType::ShipPortal:
        case GameObjectType::CubePortal:
        case GameObjectType::BallPortal:
        case GameObjectType::UfoPortal:
        case GameObjectType::WavePortal:
        case GameObjectType::RobotPortal:
        case GameObjectType::SpiderPortal:
        case GameObjectType::SwingPortal:
        case GameObjectType::RegularSizePortal:
        case GameObjectType::MiniSizePortal:
        case GameObjectType::DualPortal:
        case GameObjectType::SoloPortal:
        case GameObjectType::TeleportPortal:
        case GameObjectType::InverseMirrorPortal:
        case GameObjectType::NormalMirrorPortal:
            return env::ObjectCategory::Portal;

        case GameObjectType::Decoration:
        case GameObjectType::Modifier:
        case GameObjectType::SecretCoin:
        case GameObjectType::UserCoin:
        case GameObjectType::Collectible:
        case GameObjectType::CollisionObject:
        case GameObjectType::Special:
        case GameObjectType::EnterEffectObject:
        default:
            return env::ObjectCategory::None;
    }
}

std::vector<env::LevelObject> GameBridge::extractNearbyObjects(
    PlayLayer* playLayer,
    float playerX,
    float playerY
) {
    std::vector<env::LevelObject> result;
    if (!playLayer) return result;

    const float minX = playerX - 60.0f;
    const float maxX = playerX + 440.0f;
    const float minY = playerY - 260.0f;
    const float maxY = playerY + 260.0f;

    auto collectObject = [&](GameObject* obj) {
        if (!obj) return;
        // Filter out GD's internal player collision & anticheat proxy objects
        if (obj == playLayer->m_player1CollisionBlock ||
            obj == playLayer->m_player2CollisionBlock ||
            obj == playLayer->m_anticheatSpike) {
            return;
        }
        const auto cat = classifyObject(obj);
        if (cat == env::ObjectCategory::None) return;

        const CCRect rect = obj->getObjectRect();
        const float ox0 = rect.getMinX();
        const float ox1 = rect.getMaxX();
        const float oy0 = rect.getMinY();
        const float oy1 = rect.getMaxY();

        if (ox1 < minX || ox0 > maxX || oy1 < minY || oy0 > maxY) {
            return;
        }
        result.push_back(env::LevelObject{ox0, oy0, ox1, oy1, cat});
    };

    // Prefer GD's 2D spatial section grid (100-unit X/Y buckets) for O(1) lookup on large levels
    if (!playLayer->m_sections.empty()) {
        const int totalXSec = static_cast<int>(playLayer->m_sections.size());
        const int xSecStart = std::clamp(static_cast<int>(std::floor(minX / 100.0f)), 0, totalXSec - 1);
        const int xSecEnd = std::clamp(static_cast<int>(std::floor(maxX / 100.0f)), 0, totalXSec - 1);
        result.reserve(128);

        for (int xSec = xSecStart; xSec <= xSecEnd; ++xSec) {
            auto* col = playLayer->m_sections[static_cast<std::size_t>(xSec)];
            if (!col || col->empty()) continue;

            auto* colSizes = (static_cast<std::size_t>(xSec) < playLayer->m_sectionSizes.size())
                ? playLayer->m_sectionSizes[static_cast<std::size_t>(xSec)]
                : nullptr;

            const int totalYSec = static_cast<int>(col->size());
            const int ySecStart = std::clamp(static_cast<int>(std::floor(minY / 100.0f)), 0, totalYSec - 1);
            const int ySecEnd = std::clamp(static_cast<int>(std::floor(maxY / 100.0f)), 0, totalYSec - 1);

            for (int ySec = ySecStart; ySec <= ySecEnd; ++ySec) {
                auto* cell = (*col)[static_cast<std::size_t>(ySec)];
                if (!cell || cell->empty()) continue;

                std::size_t count = cell->size();
                if (colSizes && static_cast<std::size_t>(ySec) < colSizes->size()) {
                    const int activeCount = (*colSizes)[static_cast<std::size_t>(ySec)];
                    count = std::min(count, static_cast<std::size_t>(std::max(0, activeCount)));
                }

                for (std::size_t k = 0; k < count; ++k) {
                    collectObject((*cell)[k]);
                }
            }
        }
    } else if (playLayer->m_objects) {
        result.reserve(128);
        for (auto* obj : CCArrayExt<GameObject*>(playLayer->m_objects)) {
            collectObject(obj);
        }
    }

    return result;
}

void GameBridge::applyAction(PlayLayer* playLayer, int action, bool& currentlyHolding) {
    if (!playLayer || !playLayer->m_player1) return;
    const bool wantHold = (action != 0);
    if (wantHold == currentlyHolding) return;

    if (wantHold) {
        playLayer->m_player1->pushButton(PlayerButton::Jump);
    } else {
        playLayer->m_player1->releaseButton(PlayerButton::Jump);
    }
    currentlyHolding = wantHold;
}

void GameBridge::releaseAllInputs(PlayLayer* playLayer, bool& currentlyHolding) {
    if (currentlyHolding && playLayer && playLayer->m_player1) {
        playLayer->m_player1->releaseButton(PlayerButton::Jump);
    }
    currentlyHolding = false;
}

float GameBridge::currentPercent(PlayLayer* playLayer) {
    if (!playLayer) return 0.0f;
    return std::clamp(playLayer->getCurrentPercent(), 0.0f, 100.0f);
}

void GameBridge::setGameTimeScale(float scale) {
    if (auto* director = CCDirector::sharedDirector()) {
        if (auto* scheduler = director->getScheduler()) {
            scheduler->setTimeScale(std::clamp(scale, 0.25f, 4.0f));
        }
    }
}

} // namespace gdai::game
