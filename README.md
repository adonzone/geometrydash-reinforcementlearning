# GD-AI RL Master (Geode Mod)

A **Geode SDK v6** mod for Geometry Dash (2.2091) that trains a real **Proximal Policy Optimization (PPO)** reinforcement learning agent directly inside the game—complete with a real-time statistics HUD, live obstacle vision overlay, and a multi-tab analytics & model dashboard.

## Architecture

| Layer | Files | Purpose |
|---|---|---|
| **Pure C++23 RL Engine** | `src/rl/Tensor.hpp`, `Adam.hpp`, `RunningStats.hpp`, `PolicyNetwork.*`, `RolloutBuffer.hpp`, `PPOAgent.*` | Decoupled Actor-Critic MLP (`240 -> 128 -> 64`) with orthogonal initialization, Welford observation normalization, GAE($\lambda$), clipped PPO surrogate + clipped value loss, cosine LR schedule, entropy annealing, KL early stopping, and FNV-1a checksummed `.gdai` binary checkpoints. |
| **Environment & Telemetry** | `src/env/Observation.*`, `RewardCalculator.hpp`, `StatsTracker.*` | Translation-invariant 240D egocentric state encoder (20 player kinematics/gamemode features + 10 nearest-target precision vectors + 10×7×3 semantic grid), dense frontier-shaped rewards, ring-buffered history, and CSV export. |
| **Geometry Dash Bridge** | `src/game/GameBridge.*`, `AIManager.*`, `PlayLayerHooks.cpp`, `src/main.cpp` | Spatial-section (`m_sections`) object scanning, decoration filtering, gamemode & orb detection, clean input handling via `PlayerObject::pushButton/releaseButton`, safe training time-scaling, and auto-saving checkpoints. |
| **In-Game UI** | `src/ui/ChartNode.*`, `VisionOverlay.*`, `StatsOverlay.*`, `AIDashboardPopup.*` | Screen-locked live HUD with real-time `P(HOLD)` gauge and episode progress sparkline, world-space vision grid + target rays, and a 3-tab Geode `Popup` dashboard (`Controls`, `Analytics`, `Models & PPO`). |

## How to Build the `.geode` Mod

### Option 1: Build via Geode CLI (Local)
1. Install the [Geode CLI](https://docs.geode-sdk.org/getting-started/installation) and CMake 3.21+:
   ```bash
   brew install cmake ninja
   # Install Geode CLI & SDK (follow https://docs.geode-sdk.org/)
   geode sdk install
   ```
2. Build the mod from the repository root:
   ```bash
   geode build
   ```
   The compiled `gdai.rl-master.geode` package will be installed into your Geometry Dash Geode mods folder automatically (or found in `build/`).

### Option 2: Zero-Setup Cloud Build (GitHub Actions)
Push this repository to GitHub—the included [.github/workflows/build.yml](.github/workflows/build.yml) automatically runs the C++23 RL engine verification suite and builds a combined multi-platform `.geode` package for **Windows, macOS, iOS, Android32, and Android64**.

## Running the Standalone C++23 RL Test Suite

Because `src/rl/` and `src/env/` have zero Geode/Cocos2d dependencies, you can compile and verify the RL engine locally with any C++23 compiler:

```bash
clang++ -std=c++20 -O3 -Wall -Wextra -Wpedantic -Isrc \
    src/rl/PolicyNetwork.cpp \
    src/rl/PPOAgent.cpp \
    src/env/Observation.cpp \
    src/env/StatsTracker.cpp \
    tests/test_rl.cpp \
    -o build/gdai_tests && ./build/gdai_tests
```

Or with CMake:
```bash
cmake -B build-tests -DGDAI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-tests --parallel
ctest --test-dir build-tests --output-on-failure
```

## Using the Mod In-Game

- **Main Menu**: Click the **GD-AI** button on the bottom bar to open the Dashboard and manage saved models or inspect global stats.
- **In a Level**:
  - The **Live Stats HUD** appears in the top-right showing Episode #, Best %, Avg %, Reward, Value `V(s)`, PPO Losses/Entropy, `P(HOLD)` confidence bar, and a live progress sparkline.
  - The **Vision Overlay** renders the 10×7 egocentric obstacle grid (red = hazard, green = solid, gold = orb/pad/portal) and nearest-target rays directly in the level.
  - Click **UI** on the HUD or **AI** in the **Pause Menu** to open the full **PPO Dashboard** (`Controls`, `Analytics`, `Models & PPO`).
- **Saved Checkpoints & CSV Exports**: Stored in Geode's mod save directory under `models/master_general.gdai` (or `models/level_<id>.gdai` when Per-Level Specialization is enabled) and `stats_<id>.csv`.
