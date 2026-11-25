#include "GameBridge.hpp"

#include <algorithm>
#include <cmath>

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
            return env::ObjectCategory::Solid;

        case GameObjectType::Decoration:
            return env::ObjectCategory::None;

        default: {
            // Exclude non-interactive trigger/coin/special object types by GD enum value
            // (22=SecretCoin, 23=UserCoin, 33=Modifier/Trigger, 38=Collectible, 39=Special)
            const int rawType = static_cast<int>(obj->m_objectType);
            if (rawType == 22 || rawType == 23 || rawType == 33 || rawType == 38 || rawType == 39) {
                return env::ObjectCategory::None;
            }
            // Portals, jump pads, jump orbs/rings, gravity/size/speed changers
            return env::ObjectCategory::Interactive;
        }
    }
}

std::vector<env::LevelObject> GameBridge::extractNearbyObjects(
    PlayLayer* playLayer,
    float playerX,
    float playerY
) {
    std::vector<env::LevelObject> result;
    if (!playLayer) return result;

    const float minX = playerX - 45.0f;
    const float maxX = playerX + 380.0f;
    const float minY = playerY - 240.0f;
    const float maxY = playerY + 240.0f;

    auto collectFromArray = [&](CCArray* arr) {
        if (!arr) return;
        for (auto* obj : CCArrayExt<GameObject*>(arr)) {
            if (!obj) continue;
            const auto cat = classifyObject(obj);
            if (cat == env::ObjectCategory::None) continue;

            const CCRect rect = obj->getObjectRect();
            const float ox0 = rect.getMinX();
            const float ox1 = rect.getMaxX();
            const float oy0 = rect.getMinY();
            const float oy1 = rect.getMaxY();

            if (ox1 < minX || ox0 > maxX || oy1 < minY || oy0 > maxY) {
                continue;
            }
            result.push_back(env::LevelObject{ox0, oy0, ox1, oy1, cat});
        }
    };

    // Prefer GD's spatial section table (100-unit X buckets) for O(1) lookup on large levels
    if (playLayer->m_sections && playLayer->m_sections->count() > 0) {
        const int totalSec = static_cast<int>(playLayer->m_sections->count());
        const int secStart = std::clamp(static_cast<int>(std::floor(minX / 100.0f)), 0, totalSec - 1);
        const int secEnd = std::clamp(static_cast<int>(std::floor(maxX / 100.0f)), 0, totalSec - 1);
        result.reserve(96);
        for (int s = secStart; s <= secEnd; ++s) {
            auto* sectionArr = typeinfo_cast<CCArray*>(playLayer->m_sections->objectAtIndex(s));
            collectFromArray(sectionArr);
        }
    } else if (playLayer->m_objects) {
        result.reserve(96);
        collectFromArray(playLayer->m_objects);
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
