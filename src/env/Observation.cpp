#include "Observation.hpp"

#include <algorithm>
#include <cmath>

namespace gdai::env {

namespace {
struct HazardCandidate {
    float sortKey = 1e18f;
    float dx = 420.0f;
    float dy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
};
} // namespace

EncodedObservation ObservationEncoder::encode(
    const PlayerState& player,
    const std::vector<LevelObject>& nearbyObjects,
    float groundY
) {
    EncodedObservation out;
    out.vec.assign(EncodedObservation::kTotalDim, 0.0f);

    // 1. Player kinematics, action history, & gamemode (25 features)
    constexpr float kDegToRad = 3.1415926535f / 180.0f;
    const float rad = player.rotationDeg * kDegToRad;
    const float yAccel = player.yVelocity - player.prevYVelocity;

    std::size_t idx = 0;
    out.vec[idx++] = std::clamp((player.y - groundY - 105.0f) / 300.0f, -2.0f, 3.0f);
    out.vec[idx++] = std::clamp(player.yVelocity / 18.0f, -2.5f, 2.5f);
    out.vec[idx++] = std::clamp(yAccel / 10.0f, -2.5f, 2.5f);
    out.vec[idx++] = std::clamp(player.playerSpeed / 1.5f, 0.0f, 2.0f);
    out.vec[idx++] = std::sin(rad);
    out.vec[idx++] = std::cos(rad);
    out.vec[idx++] = player.isOnGround ? 1.0f : 0.0f;
    out.vec[idx++] = player.isUpsideDown ? 1.0f : 0.0f;
    out.vec[idx++] = player.isMini ? 1.0f : 0.0f;
    out.vec[idx++] = player.isDual ? 1.0f : 0.0f;
    out.vec[idx++] = player.isHolding ? 1.0f : 0.0f;
    out.vec[idx++] = std::clamp(player.holdDurationSec, 0.0f, 1.5f);
    out.vec[idx++] = player.touchingOrb ? 1.0f : 0.0f;
    // 4-frame action history (temporal rhythm context for UFO, Wave, Robot, Ship)
    for (float a : player.recentActions) {
        out.vec[idx++] = a;
    }

    const auto modeIdx = static_cast<std::size_t>(player.mode);
    for (std::size_t m = 0; m < static_cast<std::size_t>(Gamemode::Count); ++m) {
        out.vec[idx++] = (m == modeIdx) ? 1.0f : 0.0f;
    }

    // 2. Egocentric vision grid setup (12x8x3 = 288 cells)
    VisionGrid& grid = out.grid;
    grid.originX = player.x + VisionGrid::kOriginOffsetX;
    grid.originY = player.y + VisionGrid::kOriginOffsetY;
    grid.cells.fill(0.0f);

    const float gridMaxX = grid.originX + static_cast<float>(VisionGrid::kCols) * VisionGrid::kCellSize;
    const float gridMaxY = grid.originY + static_cast<float>(VisionGrid::kRows) * VisionGrid::kCellSize;
    const float invCell = 1.0f / VisionGrid::kCellSize;

    // Track nearest upcoming objects ahead of the player
    NearestTargets& targets = out.targets;
    targets.floorDist = std::clamp(player.y - groundY, 0.0f, 300.0f);
    targets.ceilDist = 300.0f;

    // Base ground support probes at +30, +60, +90 ahead
    const bool nearBaseGround = (player.y - groundY) <= 50.0f;
    targets.groundAhead1 = nearBaseGround ? 1.0f : 0.0f;
    targets.groundAhead2 = nearBaseGround ? 1.0f : 0.0f;
    targets.groundAhead3 = nearBaseGround ? 1.0f : 0.0f;

    std::array<HazardCandidate, 3> topHazards{};
    std::size_t hazardCount = 0;
    float bestSolidDist = 1e18f;
    float bestOrbDistSq = 1e18f;
    float bestInterDistSq = 1e18f;

    for (const auto& obj : nearbyObjects) {
        if (obj.category == ObjectCategory::None) continue;

        // Mark solid floor/ceiling distances directly around player X & ahead ground probes
        if (obj.category == ObjectCategory::Solid) {
            if (obj.minX <= player.x + 25.0f && obj.maxX >= player.x - 25.0f) {
                if (obj.maxY <= player.y + 8.0f) {
                    const float d = std::max(0.0f, player.y - obj.maxY);
                    if (d < targets.floorDist) targets.floorDist = d;
                } else if (obj.minY >= player.y - 8.0f) {
                    const float d = std::max(0.0f, obj.minY - player.y);
                    if (d < targets.ceilDist) targets.ceilDist = d;
                }
            }
            // Check whether solid platform exists beneath upcoming X probes
            if (obj.maxY <= player.y + 15.0f && obj.maxY >= player.y - 60.0f) {
                if (obj.minX <= player.x + 30.0f && obj.maxX >= player.x + 30.0f) targets.groundAhead1 = 1.0f;
                if (obj.minX <= player.x + 60.0f && obj.maxX >= player.x + 60.0f) targets.groundAhead2 = 1.0f;
                if (obj.minX <= player.x + 90.0f && obj.maxX >= player.x + 90.0f) targets.groundAhead3 = 1.0f;
            }
        }

        // Nearest-ahead precision tracking (ordered primarily by forward dx for hazards)
        const float dx = obj.centerX() - player.x;
        const float dy = obj.centerY() - player.y;
        if (dx >= -12.0f && dx <= 420.0f && std::abs(dy) <= 260.0f) {
            if (obj.category == ObjectCategory::Hazard) {
                // Prioritize hazards directly ahead in X with a gentle vertical weight
                const float sortKey = std::max(0.0f, dx) + 0.18f * std::abs(dy);
                HazardCandidate cand{sortKey, dx, dy, obj.width(), obj.height()};
                if (hazardCount < 3) {
                    topHazards[hazardCount++] = cand;
                    std::sort(topHazards.begin(), topHazards.begin() + static_cast<std::ptrdiff_t>(hazardCount),
                        [](const HazardCandidate& a, const HazardCandidate& b) { return a.sortKey < b.sortKey; });
                } else if (sortKey < topHazards[2].sortKey) {
                    topHazards[2] = cand;
                    std::sort(topHazards.begin(), topHazards.end(),
                        [](const HazardCandidate& a, const HazardCandidate& b) { return a.sortKey < b.sortKey; });
                }
            } else if (obj.category == ObjectCategory::Solid) {
                if (dx >= 0.0f) {
                    const float dist = dx + 0.35f * std::abs(dy);
                    if (dist < bestSolidDist) {
                        bestSolidDist = dist;
                        targets.hasSolid = true;
                        targets.solidDx = dx;
                        targets.solidTopDy = obj.maxY - player.y;
                        targets.solidBotDy = obj.minY - player.y;
                    }
                }
            } else if (obj.category == ObjectCategory::Orb) {
                const float distSq = dx * dx + 0.5f * dy * dy;
                if (distSq < bestOrbDistSq) {
                    bestOrbDistSq = distSq;
                    targets.hasOrb = true;
                    targets.orbDx = dx;
                    targets.orbDy = dy;
                }
                if (distSq < bestInterDistSq) {
                    bestInterDistSq = distSq;
                    targets.hasInteractive = true;
                    targets.interactiveDx = dx;
                    targets.interactiveDy = dy;
                }
            } else if (obj.category == ObjectCategory::Pad || obj.category == ObjectCategory::Portal) {
                const float distSq = dx * dx + 0.5f * dy * dy;
                if (distSq < bestInterDistSq) {
                    bestInterDistSq = distSq;
                    targets.hasInteractive = true;
                    targets.interactiveDx = dx;
                    targets.interactiveDy = dy;
                }
            }
        }

        // Rasterize object bounding box onto egocentric grid
        if (obj.maxX < grid.originX || obj.minX > gridMaxX ||
            obj.maxY < grid.originY || obj.minY > gridMaxY) {
            continue;
        }

        const int c0 = std::clamp(static_cast<int>(std::floor((obj.minX - grid.originX) * invCell)), 0, static_cast<int>(VisionGrid::kCols) - 1);
        const int c1 = std::clamp(static_cast<int>(std::floor((obj.maxX - grid.originX) * invCell)), 0, static_cast<int>(VisionGrid::kCols) - 1);
        const int r0 = std::clamp(static_cast<int>(std::floor((obj.minY - grid.originY) * invCell)), 0, static_cast<int>(VisionGrid::kRows) - 1);
        const int r1 = std::clamp(static_cast<int>(std::floor((obj.maxY - grid.originY) * invCell)), 0, static_cast<int>(VisionGrid::kRows) - 1);

        std::size_t ch = 2;
        float fillVal = 1.0f;
        if (obj.category == ObjectCategory::Hazard) {
            ch = 0;
        } else if (obj.category == ObjectCategory::Solid) {
            ch = 1;
        } else if (obj.category == ObjectCategory::Orb) {
            ch = 2;
            fillVal = 1.0f;
        } else if (obj.category == ObjectCategory::Pad) {
            ch = 2;
            fillVal = 0.75f;
        } else if (obj.category == ObjectCategory::Portal) {
            ch = 2;
            fillVal = 0.50f;
        }

        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                float& cell = grid.at(static_cast<std::size_t>(c), static_cast<std::size_t>(r), ch);
                cell = std::max(cell, fillVal);
            }
        }
    }

    if (hazardCount >= 1) {
        targets.hasHazard1 = true;
        targets.hazard1Dx = topHazards[0].dx;
        targets.hazard1Dy = topHazards[0].dy;
        targets.hazard1W = topHazards[0].w;
        targets.hazard1H = topHazards[0].h;
    }
    if (hazardCount >= 2) {
        targets.hasHazard2 = true;
        targets.hazard2Dx = topHazards[1].dx;
        targets.hazard2Dy = topHazards[1].dy;
        targets.hazard2W = topHazards[1].w;
        targets.hazard2H = topHazards[1].h;
    }
    if (hazardCount >= 3) {
        targets.hasHazard3 = true;
        targets.hazard3Dx = topHazards[2].dx;
        targets.hazard3Dy = topHazards[2].dy;
        targets.hazard3W = topHazards[2].w;
        targets.hazard3H = topHazards[2].h;
    }

    // Also mark base ground plane in solid channel when visible inside grid
    if (groundY >= grid.originY && groundY <= gridMaxY) {
        const int groundRow = std::clamp(
            static_cast<int>(std::floor((groundY - grid.originY) * invCell)),
            0,
            static_cast<int>(VisionGrid::kRows) - 1
        );
        for (int r = 0; r <= groundRow; ++r) {
            for (std::size_t c = 0; c < VisionGrid::kCols; ++c) {
                grid.at(c, static_cast<std::size_t>(r), 1) = 1.0f;
            }
        }
    }

    // 3. Append nearest-object precision features (29 features)
    // Hazard 1 (5)
    out.vec[idx++] = targets.hasHazard1 ? 1.0f : 0.0f;
    out.vec[idx++] = std::clamp(targets.hazard1Dx / 420.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.hazard1Dy / 180.0f, -1.5f, 1.5f);
    out.vec[idx++] = std::clamp(targets.hazard1W / 90.0f, 0.0f, 2.0f);
    out.vec[idx++] = std::clamp(targets.hazard1H / 90.0f, 0.0f, 2.0f);

    // Hazard 2 (5)
    out.vec[idx++] = targets.hasHazard2 ? 1.0f : 0.0f;
    out.vec[idx++] = std::clamp(targets.hazard2Dx / 420.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.hazard2Dy / 180.0f, -1.5f, 1.5f);
    out.vec[idx++] = std::clamp(targets.hazard2W / 90.0f, 0.0f, 2.0f);
    out.vec[idx++] = std::clamp(targets.hazard2H / 90.0f, 0.0f, 2.0f);

    // Hazard 3 (4)
    out.vec[idx++] = targets.hasHazard3 ? 1.0f : 0.0f;
    out.vec[idx++] = std::clamp(targets.hazard3Dx / 420.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.hazard3Dy / 180.0f, -1.5f, 1.5f);
    out.vec[idx++] = std::clamp(targets.hazard3H / 90.0f, 0.0f, 2.0f);

    // Nearest Solid platform (4)
    out.vec[idx++] = targets.hasSolid ? 1.0f : 0.0f;
    out.vec[idx++] = std::clamp(targets.solidDx / 420.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.solidTopDy / 180.0f, -1.5f, 1.5f);
    out.vec[idx++] = std::clamp(targets.solidBotDy / 180.0f, -1.5f, 1.5f);

    // Nearest Orb (3)
    out.vec[idx++] = targets.hasOrb ? 1.0f : 0.0f;
    out.vec[idx++] = std::clamp(targets.orbDx / 420.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.orbDy / 180.0f, -1.5f, 1.5f);

    // Nearest Pad/Portal/Interactive (3)
    out.vec[idx++] = targets.hasInteractive ? 1.0f : 0.0f;
    out.vec[idx++] = std::clamp(targets.interactiveDx / 420.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.interactiveDy / 180.0f, -1.5f, 1.5f);

    // Vertical clearance & ahead ground probes (5)
    out.vec[idx++] = std::clamp(targets.floorDist / 180.0f, 0.0f, 2.0f);
    out.vec[idx++] = std::clamp(targets.ceilDist / 180.0f, 0.0f, 2.0f);
    out.vec[idx++] = targets.groundAhead1;
    out.vec[idx++] = targets.groundAhead2;
    out.vec[idx++] = targets.groundAhead3;

    // 4. Append flattened grid (288 features)
    std::copy(grid.cells.begin(), grid.cells.end(), out.vec.begin() + static_cast<std::ptrdiff_t>(idx));

    return out;
}

} // namespace gdai::env
