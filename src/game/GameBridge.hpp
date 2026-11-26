#pragma once

#include "../env/Observation.hpp"

#include <Geode/Geode.hpp>
#include <vector>

namespace gdai::game {

/// Translates live Geometry Dash PlayLayer / PlayerObject / GameObject state into RL structs.
class GameBridge {
public:
    static env::PlayerState extractPlayerState(
        PlayLayer* playLayer,
        bool aiHolding,
        float holdDurationSec
    );

    static std::vector<env::LevelObject> extractNearbyObjects(
        PlayLayer* playLayer,
        float playerX,
        float playerY
    );

    static env::ObjectCategory classifyObject(GameObject* obj);

    enum class ActionType : int {
        Release = 0,
        Click = 1,
        Hold = 2
    };

    static void applyAction(PlayLayer* playLayer, int action, bool& currentlyHolding);
    static void releaseAllInputs(PlayLayer* playLayer, bool& currentlyHolding);

    static float currentPercent(PlayLayer* playLayer);
    static void setGameTimeScale(float scale);
};

} // namespace gdai::game
