#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gdai::env {

enum class Gamemode : std::uint8_t {
    Cube = 0,
    Ship = 1,
    Ball = 2,
    UFO = 3,
    Wave = 4,
    Robot = 5,
    Spider = 6,
    Swing = 7,
    Count = 8
};

inline const char* gamemodeName(Gamemode mode) {
    switch (mode) {
        case Gamemode::Cube:   return "Cube";
        case Gamemode::Ship:   return "Ship";
        case Gamemode::Ball:   return "Ball";
        case Gamemode::UFO:    return "UFO";
        case Gamemode::Wave:   return "Wave";
        case Gamemode::Robot:  return "Robot";
        case Gamemode::Spider: return "Spider";
        case Gamemode::Swing:  return "Swing";
        default:               return "Unknown";
    }
}

enum class ObjectCategory : std::uint8_t {
    None = 0,
    Hazard = 1,      // Spikes, saws, lethal blocks
    Solid = 2,       // Walkable blocks, slopes, platforms
    Interactive = 3  // Jump orbs, jump pads, portals
};

struct LevelObject {
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;
    ObjectCategory category = ObjectCategory::None;

    float centerX() const { return 0.5f * (minX + maxX); }
    float centerY() const { return 0.5f * (minY + maxY); }
};

struct PlayerState {
    float x = 0.0f;
    float y = 0.0f;
    float yVelocity = 0.0f;
    float playerSpeed = 0.9f;
    float rotationDeg = 0.0f;
    bool isOnGround = true;
    bool isUpsideDown = false;
    bool isMini = false;
    bool isDual = false;
    bool isHolding = false;
    float holdDurationSec = 0.0f;
    bool touchingOrb = false;
    Gamemode mode = Gamemode::Cube;
};

struct VisionGrid {
    static constexpr std::size_t kCols = 10;
    static constexpr std::size_t kRows = 7;
    static constexpr std::size_t kChannels = 3; // 0=Hazard, 1=Solid, 2=Interactive
    static constexpr float kCellSize = 30.0f;   // 1 standard Geometry Dash grid block
    static constexpr float kOriginOffsetX = -15.0f;
    static constexpr float kOriginOffsetY = -105.0f; // 3.5 blocks below player

    float originX = 0.0f;
    float originY = 0.0f;
    std::array<float, kCols * kRows * kChannels> cells{};

    float at(std::size_t col, std::size_t row, std::size_t ch) const {
        return cells[(row * kCols + col) * kChannels + ch];
    }

    float& at(std::size_t col, std::size_t row, std::size_t ch) {
        return cells[(row * kCols + col) * kChannels + ch];
    }
};

struct NearestTargets {
    bool hasHazard1 = false;
    float hazard1Dx = 360.0f, hazard1Dy = 0.0f;

    bool hasHazard2 = false;
    float hazard2Dx = 360.0f, hazard2Dy = 0.0f;

    bool hasSolid = false;
    float solidDx = 360.0f, solidDy = 0.0f;

    bool hasInteractive = false;
    float interactiveDx = 360.0f, interactiveDy = 0.0f;

    float floorDist = 180.0f;
    float ceilDist = 180.0f;
};

struct EncodedObservation {
    static constexpr std::size_t kPlayerFeatures = 20;
    static constexpr std::size_t kNearestFeatures = 10;
    static constexpr std::size_t kGridFeatures = VisionGrid::kCols * VisionGrid::kRows * VisionGrid::kChannels; // 210
    static constexpr std::size_t kTotalDim = kPlayerFeatures + kNearestFeatures + kGridFeatures;                // 240

    std::vector<float> vec;
    VisionGrid grid;
    NearestTargets targets;
};

/// Encodes Geometry Dash player + nearby level geometry into a translation-invariant 240D vector.
class ObservationEncoder {
public:
    static EncodedObservation encode(
        const PlayerState& player,
        const std::vector<LevelObject>& nearbyObjects,
        float groundY = 0.0f
    );
};

} // namespace gdai::env
