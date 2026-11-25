#include "Observation.hpp"

#include <algorithm>
#include <cmath>

namespace gdai::env {

EncodedObservation ObservationEncoder::encode(
    const PlayerState& player,
    const std::vector<LevelObject>& nearbyObjects,
    float groundY
) {
    EncodedObservation out;
    out.vec.assign(EncodedObservation::kTotalDim, 0.0f);

    // 1. Player kinematics & gamemode (20 features)
    constexpr float kDegToRad = 3.1415926535f / 180.0f;
    const float rad = player.rotationDeg * kDegToRad;

    std::size_t idx = 0;
    out.vec[idx++] = std::clamp((player.y - groundY - 105.0f) / 300.0f, -2.0f, 3.0f);
    out.vec[idx++] = std::clamp(player.yVelocity / 18.0f, -2.5f, 2.5f);
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

    const auto modeIdx = static_cast<std::size_t>(player.mode);
    for (std::size_t m = 0; m < static_cast<std::size_t>(Gamemode::Count); ++m) {
        out.vec[idx++] = (m == modeIdx) ? 1.0f : 0.0f;
    }

    // 2. Egocentric vision grid setup
    VisionGrid& grid = out.grid;
    grid.originX = player.x + VisionGrid::kOriginOffsetX;
    grid.originY = player.y + VisionGrid::kOriginOffsetY;
    grid.cells.fill(0.0f);

    const float gridMaxX = grid.originX + static_cast<float>(VisionGrid::kCols) * VisionGrid::kCellSize;
    const float gridMaxY = grid.originY + static_cast<float>(VisionGrid::kRows) * VisionGrid::kCellSize;
    const float invCell = 1.0f / VisionGrid::kCellSize;

    // Track nearest objects ahead of the player
    NearestTargets& targets = out.targets;
    targets.floorDist = std::clamp(player.y - groundY, 0.0f, 300.0f);
    targets.ceilDist = 300.0f;

    float bestHaz1DistSq = 1e18f;
    float bestHaz2DistSq = 1e18f;
    float bestSolidDistSq = 1e18f;
    float bestInterDistSq = 1e18f;

    for (const auto& obj : nearbyObjects) {
        if (obj.category == ObjectCategory::None) continue;

        // Mark solid ground/ceiling distances directly around player X
        if (obj.category == ObjectCategory::Solid && obj.minX <= player.x + 25.0f && obj.maxX >= player.x - 25.0f) {
            if (obj.maxY <= player.y + 5.0f) {
                const float d = std::max(0.0f, player.y - obj.maxY);
                if (d < targets.floorDist) targets.floorDist = d;
            } else if (obj.minY >= player.y - 5.0f) {
                const float d = std::max(0.0f, obj.minY - player.y);
                if (d < targets.ceilDist) targets.ceilDist = d;
            }
        }

        // Nearest-ahead tracking
        const float dx = obj.centerX() - player.x;
        const float dy = obj.centerY() - player.y;
        if (dx >= -15.0f && dx <= 360.0f && std::abs(dy) <= 240.0f) {
            const float distSq = dx * dx + 0.5f * dy * dy;
            if (obj.category == ObjectCategory::Hazard) {
                if (distSq < bestHaz1DistSq) {
                    bestHaz2DistSq = bestHaz1DistSq;
                    targets.hasHazard2 = targets.hasHazard1;
                    targets.hazard2Dx = targets.hazard1Dx;
                    targets.hazard2Dy = targets.hazard1Dy;

                    bestHaz1DistSq = distSq;
                    targets.hasHazard1 = true;
                    targets.hazard1Dx = dx;
                    targets.hazard1Dy = dy;
                } else if (distSq < bestHaz2DistSq) {
                    bestHaz2DistSq = distSq;
                    targets.hasHazard2 = true;
                    targets.hazard2Dx = dx;
                    targets.hazard2Dy = dy;
                }
            } else if (obj.category == ObjectCategory::Solid) {
                if (dx >= 0.0f && distSq < bestSolidDistSq) {
                    bestSolidDistSq = distSq;
                    targets.hasSolid = true;
                    targets.solidDx = dx;
                    targets.solidDy = dy;
                }
            } else if (obj.category == ObjectCategory::Interactive) {
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

        const std::size_t ch = static_cast<std::size_t>(obj.category) - 1; // 0=Hazard, 1=Solid, 2=Interactive
        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                grid.at(static_cast<std::size_t>(c), static_cast<std::size_t>(r), ch) = 1.0f;
            }
        }
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

    // 3. Append nearest-object precision features (10 features)
    out.vec[idx++] = std::clamp(targets.hazard1Dx / 360.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.hazard1Dy / 180.0f, -1.5f, 1.5f);
    out.vec[idx++] = std::clamp(targets.hazard2Dx / 360.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.hazard2Dy / 180.0f, -1.5f, 1.5f);
    out.vec[idx++] = std::clamp(targets.solidDx / 360.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.solidDy / 180.0f, -1.5f, 1.5f);
    out.vec[idx++] = std::clamp(targets.interactiveDx / 360.0f, -0.2f, 1.0f);
    out.vec[idx++] = std::clamp(targets.interactiveDy / 180.0f, -1.5f, 1.5f);
    out.vec[idx++] = std::clamp(targets.floorDist / 180.0f, 0.0f, 2.0f);
    out.vec[idx++] = std::clamp(targets.ceilDist / 180.0f, 0.0f, 2.0f);

    // 4. Append flattened grid (210 features)
    std::copy(grid.cells.begin(), grid.cells.end(), out.vec.begin() + static_cast<std::ptrdiff_t>(idx));

    return out;
}

} // namespace gdai::env
