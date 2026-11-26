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

inline const char* gamemodeSlug(Gamemode mode) {
    switch (mode) {
        case Gamemode::Cube:   return "cube";
        case Gamemode::Ship:   return "ship";
        case Gamemode::Ball:   return "ball";
        case Gamemode::UFO:    return "ufo";
        case Gamemode::Wave:   return "wave";
        case Gamemode::Robot:  return "robot";
        case Gamemode::Spider: return "spider";
        case Gamemode::Swing:  return "swing";
        default:               return "unknown";
    }
}

enum class ObjectCategory : std::uint8_t {
    None = 0,
    Hazard = 1,      // Spikes, saws, lethal blocks
    Solid = 2,       // Walkable blocks, slopes, breakables, platforms
    Orb = 3,         // Jump rings / orbs (interactive on press)
    Pad = 4,         // Jump / gravity / spider pads (auto-launch on touch)
    Portal = 5,      // Gamemode, gravity, size, speed, dual, teleport portals
    Interactive = 3  // Legacy alias for Orb/interactive checks
};

struct LevelObject {
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;
    ObjectCategory category = ObjectCategory::None;

    float centerX() const { return 0.5f * (minX + maxX); }
    float centerY() const { return 0.5f * (minY + maxY); }
    float width() const { return maxX - minX; }
    float height() const { return maxY - minY; }
};

struct PlayerState {
    float x = 0.0f;
    float y = 0.0f;
    float yVelocity = 0.0f;
    float prevYVelocity = 0.0f;
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
    std::array<float, 4> recentActions{0.0f, 0.0f, 0.0f, 0.0f}; // newest first
};

struct VisionGrid {
    static constexpr std::size_t kCols = 12;
    static constexpr std::size_t kRows = 8;
    static constexpr std::size_t kChannels = 3; // 0=Hazard, 1=Solid, 2=Orb/Pad/Portal
    static constexpr float kCellSize = 30.0f;   // 1 standard Geometry Dash grid block
    static constexpr float kOriginOffsetX = -30.0f;  // 1 block behind -> 11 blocks (330u) ahead
    static constexpr float kOriginOffsetY = -120.0f; // 4 blocks below -> 4 blocks above

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
    float hazard1Dx = 420.0f, hazard1Dy = 0.0f, hazard1W = 0.0f, hazard1H = 0.0f;

    bool hasHazard2 = false;
    float hazard2Dx = 420.0f, hazard2Dy = 0.0f, hazard2W = 0.0f, hazard2H = 0.0f;

    bool hasHazard3 = false;
    float hazard3Dx = 420.0f, hazard3Dy = 0.0f, hazard3W = 0.0f, hazard3H = 0.0f;

    bool hasSolid = false;
    float solidDx = 420.0f, solidTopDy = 0.0f, solidBotDy = 0.0f;

    bool hasOrb = false;
    float orbDx = 420.0f, orbDy = 0.0f;

    bool hasInteractive = false; // Nearest Pad or Portal (or Orb fallback for overlay)
    float interactiveDx = 420.0f, interactiveDy = 0.0f;

    float floorDist = 180.0f;
    float ceilDist = 180.0f;

    // Ground/solid support probes at +30, +60, +90 world units ahead
    float groundAhead1 = 0.0f;
    float groundAhead2 = 0.0f;
    float groundAhead3 = 0.0f;
};

struct EncodedObservation {
    static constexpr std::size_t kPlayerFeatures = 25;
    static constexpr std::size_t kNearestFeatures = 29;
    static constexpr std::size_t kGridFeatures = VisionGrid::kCols * VisionGrid::kRows * VisionGrid::kChannels; // 288
    static constexpr std::size_t kContinuousNormDim = kPlayerFeatures + kNearestFeatures;                        // 54
    static constexpr std::size_t kTotalDim = kPlayerFeatures + kNearestFeatures + kGridFeatures;                 // 342

    std::vector<float> vec;
    VisionGrid grid;
    NearestTargets targets;
};

/// Encodes Geometry Dash player + nearby level geometry into a translation-invariant 342D vector.
class ObservationEncoder {
public:
    static EncodedObservation encode(
        const PlayerState& player,
        const std::vector<LevelObject>& nearbyObjects,
        float groundY = 0.0f
    );
};

} // namespace gdai::env
