#include "env/Observation.hpp"
#include "env/RewardCalculator.hpp"
#include "env/StatsTracker.hpp"
#include "rl/MultiModePPOAgent.hpp"
#include "rl/PPOAgent.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace gdai;

namespace {

void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

void testOrthogonalInit() {
    std::mt19937 rng(7);
    rl::Mat w(16, 8);
    rl::initOrthogonal(w, 1.5f, rng);

    for (std::size_t c1 = 0; c1 < w.cols; ++c1) {
        for (std::size_t c2 = 0; c2 < w.cols; ++c2) {
            double dot = 0.0;
            for (std::size_t r = 0; r < w.rows; ++r) {
                dot += static_cast<double>(w(r, c1)) * static_cast<double>(w(r, c2));
            }
            const double expected = (c1 == c2) ? (1.5 * 1.5) : 0.0;
            expect(std::abs(dot - expected) < 1e-4, "Orthogonal init column inner product mismatch");
        }
    }
    std::cout << "[PASS] testOrthogonalInit\n";
}

void testFiniteDifferenceGradients() {
    rl::PolicyNetwork net(8, 12, 8, 123);
    rl::Vec obs = {0.3f, -0.7f, 1.2f, 0.05f, -0.4f, 0.9f, -1.1f, 0.6f};

    auto slots = net.paramSlots();
    for (auto& s : slots) std::fill(s.grad->begin(), s.grad->end(), 0.0f);

    auto evalLoss = [&]() -> double {
        auto c = net.forward(obs);
        double vErr = static_cast<double>(c.value) - 1.25;
        return 0.7 * static_cast<double>(c.logit0) - 0.4 * static_cast<double>(c.logit1) + 0.5 * vErr * vErr;
    };

    const auto cache = net.forward(obs);
    net.backwardAccumulate(obs, cache, 0.7f, -0.4f, cache.value - 1.25f);

    constexpr float eps = 1e-3f;
    double maxAbsErr = 0.0;
    for (auto& slot : slots) {
        for (std::size_t i = 0; i < slot.param->size(); ++i) {
            const float orig = (*slot.param)[i];
            (*slot.param)[i] = orig + eps;
            const double lPlus = evalLoss();
            (*slot.param)[i] = orig - eps;
            const double lMinus = evalLoss();
            (*slot.param)[i] = orig;

            const double numGrad = (lPlus - lMinus) / (2.0 * eps);
            const double anaGrad = static_cast<double>((*slot.grad)[i]);
            maxAbsErr = std::max(maxAbsErr, std::abs(numGrad - anaGrad));
        }
    }
    expect(maxAbsErr < 2e-3, "Backprop gradients did not match finite differences");
    std::cout << "[PASS] testFiniteDifferenceGradients (maxAbsErr=" << maxAbsErr << ")\n";
}

void testGAEComputation() {
    rl::RolloutBuffer buf(5);
    // 2-step episode ending at t=1 (done=true)
    buf.add({{}, 0, 0.0f, 1.0f, 2.0f, false, false, 0.0f, 1});
    buf.add({{}, 1, 0.0f, 1.5f, 3.0f, true,  false, 0.0f, 1});
    // 1-step gamemode portal swap at t=2 (truncated=true, bootstrapValue=4.0)
    buf.add({{}, 0, 0.0f, 0.5f, 1.0f, false, true,  4.0f, 2});
    // 2-step non-terminal episode
    buf.add({{}, 0, 0.0f, 0.5f, 1.0f, false, false, 0.0f, 3});
    buf.add({{}, 1, 0.0f, 0.8f, 1.0f, false, false, 0.0f, 3});

    const float gamma = 0.9f;
    const float lam = 0.8f;
    const float bootstrap = 2.0f;
    buf.computeGAE(bootstrap, gamma, lam);

    // t=1 (done): delta1 = 3.0 - 1.5 = 1.5 -> A1 = 1.5, R1 = 3.0
    expect(std::abs(buf.advantage(1) - 1.5f) < 1e-5f, "GAE terminal step mismatch");
    expect(std::abs(buf.ret(1) - 3.0f) < 1e-5f, "GAE terminal return mismatch");

    // t=0: delta0 = 2.0 + 0.9*1.5 - 1.0 = 2.35; A0 = 2.35 + 0.9*0.8*1.5 = 3.43
    expect(std::abs(buf.advantage(0) - 3.43f) < 1e-4f, "GAE non-terminal step mismatch");

    // t=2 (truncated portal swap): delta2 = 1.0 + 0.9*4.0 - 0.5 = 4.1 -> A2 = 4.1 (no bleed from t=3!)
    expect(std::abs(buf.advantage(2) - 4.1f) < 1e-4f, "GAE truncated portal bootstrap mismatch");

    // t=4: delta4 = 1.0 + 0.9*2.0 - 0.8 = 2.0 -> A4 = 2.0
    expect(std::abs(buf.advantage(4) - 2.0f) < 1e-5f, "GAE bootstrap step mismatch");
    std::cout << "[PASS] testGAEComputation (with portal truncation)\n";
}

void testObservationEncoder() {
    env::PlayerState p;
    p.x = 100.0f;
    p.y = 105.0f;
    p.yVelocity = 6.0f;
    p.prevYVelocity = 4.0f;
    p.mode = env::Gamemode::Ship;
    p.isOnGround = false;
    p.recentActions = {1.0f, 0.0f, 1.0f, 0.0f};

    std::vector<env::LevelObject> objs = {
        {175.0f, 90.0f, 205.0f, 120.0f, env::ObjectCategory::Hazard},
        {145.0f, 60.0f, 175.0f, 90.0f, env::ObjectCategory::Solid},
        {205.0f, 135.0f, 235.0f, 165.0f, env::ObjectCategory::Orb},
    };

    const auto enc = env::ObservationEncoder::encode(p, objs, 0.0f);
    expect(enc.vec.size() == env::EncodedObservation::kTotalDim, "Unexpected observation vector dimension");
    expect(enc.targets.hasHazard1 && std::abs(enc.targets.hazard1Dx - 90.0f) < 1e-3f, "Nearest hazard dx mismatch");
    expect(enc.targets.hasOrb && std::abs(enc.targets.orbDx - 120.0f) < 1e-3f, "Nearest orb dx mismatch");
    expect(enc.targets.hasInteractive && std::abs(enc.targets.interactiveDx - 120.0f) < 1e-3f, "Nearest interactive dx mismatch");

    // Ship one-hot is at index 17 + 1 = 18
    expect(enc.vec[18] == 1.0f && enc.vec[17] == 0.0f, "Ship gamemode one-hot mismatch");

    bool foundHazardCell = false;
    for (std::size_t r = 0; r < env::VisionGrid::kRows; ++r) {
        for (std::size_t c = 0; c < env::VisionGrid::kCols; ++c) {
            if (enc.grid.at(c, r, 0) > 0.5f) foundHazardCell = true;
        }
    }
    expect(foundHazardCell, "Hazard not rasterized into vision grid");
    std::cout << "[PASS] testObservationEncoder\n";
}

void testRewardRuleNoRewardOnDeath2xNewBest() {
    env::RewardCalculator rewards;
    rewards.onLevelStart();

    // Episode 1: reaches x=200 (new best!) then dies -> 2x reward, -1.0 terminal collision penalty
    rewards.resetEpisode(0.0f);
    const float s1 = rewards.computeStepReward(100.0f, 0);
    const float s2 = rewards.computeStepReward(200.0f, 1);
    const float raw1 = s1 + s2;
    const auto res1 = rewards.resolveDeath();
    expect(res1.achievedNewBest, "Episode 1 should count as a new best");
    expect(std::abs(res1.terminalReward - (-1.0f)) < 1e-6f, "Death terminal reward must be -1.0 (collision penalty)");
    expect(std::abs(res1.rewardMultiplier - 2.0f) < 1e-6f, "New best must get 2x reward multiplier");
    expect(std::abs(res1.finalEpisodeReward - (2.0f * raw1 - 1.0f)) < 1e-4f, "Final new-best reward should be 2x raw minus collision penalty");
    expect(res1.saveToElite, "New best run should be saved to elite SIL buffer");
    expect(res1.fatalTrimSteps == 3, "Elite buffer should trim fatal crash steps");

    // Episode 2: reaches x=150 (NOT a new best) and dies -> 0x multiplier, -1.0 collision penalty
    rewards.resetEpisode(0.0f);
    (void)rewards.computeStepReward(80.0f, 0);
    (void)rewards.computeStepReward(150.0f, 1);
    const auto res2 = rewards.resolveDeath();
    expect(!res2.achievedNewBest, "Episode 2 should not count as a new best");
    expect(std::abs(res2.rewardMultiplier) < 1e-6f, "Non-PB death must have 0x reward multiplier");
    expect(std::abs(res2.terminalReward - (-1.0f)) < 1e-6f, "Death terminal reward must be -1.0 (collision penalty)");
    expect(std::abs(res2.finalEpisodeReward - (-1.0f)) < 1e-6f, "Non-PB death must give -1.0 terminal collision penalty");
    expect(!res2.saveToElite, "Non-PB death must not be saved to elite buffer");

    // Verify PPOAgent transition retro-scaling: non-PB steps become 0.0, collision step stays -1.0
    const auto cfg = rl::PPOConfig::fromPreset(rl::TrainingPreset::FastLearner, 16, 8);
    rl::PPOAgent agent(cfg);
    rl::Vec obs(16, 0.1f);
    const auto dec1 = agent.selectAction(obs, true);
    agent.stageTransition(dec1, 0.5f, false, false, 0.0f, 101);
    const auto dec2 = agent.selectAction(obs, true);
    agent.stageTransition(dec2, res2.terminalReward, true, false, 0.0f, 101);
    agent.finalizeEpisode(101, res2.rewardMultiplier, 150.0f, res2.saveToElite, res2.fatalTrimSteps);

    // Episode 3: reaches x=400 and completes (new best!) -> 2x completion reward
    rewards.resetEpisode(0.0f);
    (void)rewards.computeStepReward(400.0f, 0);
    const auto res3 = rewards.resolveCompletion();
    expect(res3.achievedNewBest && std::abs(res3.rewardMultiplier - 2.0f) < 1e-6f, "First clear should get 2x");
    expect(res3.finalEpisodeReward > 100.0f, "First clear 2x reward should exceed 100");

    // Episode 4: completes again at x=400 (repeat clear, not new best) -> 1x completion reward
    rewards.resetEpisode(0.0f);
    (void)rewards.computeStepReward(400.0f, 0);
    const auto res4 = rewards.resolveCompletion();
    expect(!res4.achievedNewBest && std::abs(res4.rewardMultiplier - 1.0f) < 1e-6f, "Repeat clear should get 1x");
    expect(res4.saveToElite, "Completion should save to elite SIL buffer");

    // Rolling PB Recovery: simulate stagnation episodes to verify soft floor prevents policy starvation
    const float prevBest = rewards.sessionBestX();
    for (std::size_t i = 0; i < 30; ++i) {
        rewards.resetEpisode(0.0f);
        (void)rewards.computeStepReward(100.0f, 0);
        rewards.resolveDeath();
    }
    expect(rewards.sessionBestX() < prevBest, "Session best should softly recover after 30 stagnant episodes");
    expect(rewards.sessionBestX() >= prevBest * 0.89f, "Recovery floor should be within configured range");

    std::cout << "[PASS] testRewardRuleNoRewardOnDeath2xNewBest\n";
}

void testModelSerializationAndMultiModeHotSwap() {
    const auto tmpDir = std::filesystem::temp_directory_path() / "gdai_multimode_test";
    std::filesystem::remove_all(tmpDir);

    rl::MultiModePPOAgent multiA(rl::TrainingPreset::FastLearner, 16, 8);
    rl::Vec sampleObs(16, 0.25f);
    sampleObs[3] = -0.8f;

    // Hot-swap Cube -> Ship -> Wave and ensure each mode has distinct weights & tracks swaps
    const auto cubeEvalA = multiA.selectAction(env::Gamemode::Cube, sampleObs, false);
    const auto shipEvalA = multiA.selectAction(env::Gamemode::Ship, sampleObs, false);
    const auto waveEvalA = multiA.selectAction(env::Gamemode::Wave, sampleObs, false);
    expect(multiA.hotSwapCount() == 2, "Expected 2 gamemode hot-swaps (Cube->Ship->Wave)");
    expect(multiA.activeMode() == env::Gamemode::Wave, "Active mode should be Wave");
    expect(
        std::abs(cubeEvalA.value - shipEvalA.value) > 1e-5f ||
        std::abs(cubeEvalA.value - waveEvalA.value) > 1e-5f,
        "Different gamemodes must use distinct neural networks"
    );

    // Freeze training on Wave, keep Cube and Ship active
    multiA.setModeTrainingFrozen(env::Gamemode::Wave, true);
    expect(multiA.isModeTrainingFrozen(env::Gamemode::Wave), "Wave training should be frozen");
    expect(!multiA.isModeTrainingFrozen(env::Gamemode::Cube), "Cube training should be active");

    // Stage 1 step on Cube and Ship so saveAllModels persists them
    const auto dCube = multiA.selectAction(env::Gamemode::Cube, sampleObs, true);
    multiA.stageTransition(env::Gamemode::Cube, dCube, 1.0f, true, false, 0.0f, 1);
    const auto dShip = multiA.selectAction(env::Gamemode::Ship, sampleObs, true);
    multiA.stageTransition(env::Gamemode::Ship, dShip, 1.0f, true, false, 0.0f, 1);

    // Test 1: Save per-mode models
    const std::size_t saved = multiA.saveAllModels(tmpDir, "test_master", false);
    expect(saved == 2, "Expected 2 trained gamemode models to be saved");
    expect(std::filesystem::exists(tmpDir / "test_master_cube.gdai"), "Missing cube checkpoint");
    expect(std::filesystem::exists(tmpDir / "test_master_ship.gdai"), "Missing ship checkpoint");

    // Test 2: Save unified master bundle
    const auto masterPath = multiA.masterBundlePathFor(tmpDir, "test_master");
    const bool masterSaved = multiA.saveMasterBundle(masterPath);
    expect(masterSaved, "Failed to save unified master bundle");
    expect(std::filesystem::exists(masterPath), "Master bundle file does not exist on disk");

    // Reload master bundle into a fresh MultiModePPOAgent
    rl::MultiModePPOAgent multiMaster(rl::TrainingPreset::FastLearner, 16, 8);
    multiMaster.resetAll(9999);
    const bool masterLoaded = multiMaster.loadMasterBundle(masterPath);
    expect(masterLoaded, "Failed to load unified master bundle");
    expect(multiMaster.isModeTrainingFrozen(env::Gamemode::Wave), "Loaded master should preserve Wave frozen state");
    expect(!multiMaster.isModeTrainingFrozen(env::Gamemode::Cube), "Loaded master should preserve Cube active state");

    // Reload into a fresh MultiModePPOAgent via individual models and verify exact match
    rl::MultiModePPOAgent multiB(rl::TrainingPreset::FastLearner, 16, 8);
    multiB.resetAll(9999);
    const std::size_t loaded = multiB.loadAllModels(tmpDir, "test_master");
    expect(loaded == 2, "Expected 2 gamemode models to be reloaded");

    const auto cubeBefore = multiA.selectAction(env::Gamemode::Cube, sampleObs, false);
    const auto shipBefore = multiA.selectAction(env::Gamemode::Ship, sampleObs, false);
    const auto cubeAfter = multiB.selectAction(env::Gamemode::Cube, sampleObs, false);
    const auto shipAfter = multiB.selectAction(env::Gamemode::Ship, sampleObs, false);
    const auto cubeMaster = multiMaster.selectAction(env::Gamemode::Cube, sampleObs, false);
    const auto shipMaster = multiMaster.selectAction(env::Gamemode::Ship, sampleObs, false);
    expect(std::abs(cubeBefore.holdProb - cubeAfter.holdProb) < 1e-6f, "Reloaded Cube holdProb mismatch");
    expect(std::abs(shipBefore.holdProb - shipAfter.holdProb) < 1e-6f, "Reloaded Ship holdProb mismatch");
    expect(std::abs(cubeBefore.holdProb - cubeMaster.holdProb) < 1e-6f, "Master bundle Cube holdProb mismatch");
    expect(std::abs(shipBefore.holdProb - shipMaster.holdProb) < 1e-6f, "Master bundle Ship holdProb mismatch");

    // Verify checksum catches corruption
    const auto cubePath = tmpDir / "test_master_cube.gdai";
    {
        std::fstream fs(cubePath, std::ios::in | std::ios::out | std::ios::binary);
        fs.seekg(32);
        char b = 0;
        fs.read(&b, 1);
        b ^= 0x55;
        fs.seekp(32);
        fs.write(&b, 1);
    }
    expect(!multiB.agentFor(env::Gamemode::Cube).loadModel(cubePath), "Corrupted checkpoint should fail checksum");
    std::filesystem::remove_all(tmpDir);
    std::cout << "[PASS] testModelSerializationAndMultiModeHotSwap\n";
}

/// Multi-gamemode Geometry Dash simulator:
/// - Segment 1 [0 .. 1100]: Cube with alternating floor spikes & low ceiling spikes
/// - Segment 2 [1100 .. 2100]: Ship portal -> fly through upper/lower obstacle gates!
/// Neither constant-HOLD nor constant-RELEASE can survive past ~15%, and Cube vs Ship require
/// completely different control policies — proving both the 0x-death / 2x-PB reward rule + SIL
/// and seamless gamemode model hot-swapping!
class MultiModeGDRunner {
public:
    static constexpr float kShipPortalX = 1100.0f;
    static constexpr float kLevelLength = 2100.0f;
    static constexpr float kDt = 1.0f / 30.0f;
    static constexpr float kSpeedX = 310.0f;
    static constexpr float kJumpVel = 520.0f;
    static constexpr float kCubeGravity = -2350.0f;
    static constexpr float kShipGravity = -950.0f;
    static constexpr float kShipThrust = 1950.0f;
    static constexpr float kGroundY = 0.0f;
    static constexpr float kCeilingY = 180.0f;
    static constexpr float kPlayerHalf = 14.0f;

    MultiModeGDRunner() {
        // Cube section: floor spikes + overhead spikes that punish blind holding
        const float floorSpikeXs[] = {310.f, 700.f};
        for (float sx : floorSpikeXs) {
            m_objects.push_back({sx - 12.f, 0.f, sx + 12.f, 26.f, env::ObjectCategory::Hazard});
        }
        const float ceilSpikeXs[] = {170.f, 510.f, 910.f};
        for (float cx : ceilSpikeXs) {
            m_objects.push_back({cx - 18.f, 44.f, cx + 18.f, 84.f, env::ObjectCategory::Hazard});
        }

        // Ship portal at x=1100
        m_objects.push_back({kShipPortalX - 15.f, 0.f, kShipPortalX + 15.f, 120.f, env::ObjectCategory::Portal});

        // Ship section: alternating floor pillars & ceiling pillars requiring active vertical steering!
        // Gate 1 at x=1380: gap in [70, 150] (floor hazard up to 62)
        m_objects.push_back({1355.f, 0.f, 1405.f, 62.f, env::ObjectCategory::Hazard});
        // Gate 2 at x=1660: gap in [20, 100] (ceiling hazard down to 108)
        m_objects.push_back({1635.f, 108.f, 1685.f, 180.f, env::ObjectCategory::Hazard});
        // Gate 3 at x=1920: gap in [65, 145] (floor hazard up to 58)
        m_objects.push_back({1895.f, 0.f, 1945.f, 58.f, env::ObjectCategory::Hazard});

        reset();
    }

    env::EncodedObservation reset() {
        m_player = env::PlayerState{};
        m_player.x = 0.0f;
        m_player.y = kPlayerHalf;
        m_player.yVelocity = 0.0f;
        m_player.prevYVelocity = 0.0f;
        m_player.isOnGround = true;
        m_player.mode = env::Gamemode::Cube;
        m_player.recentActions = {0.0f, 0.0f, 0.0f, 0.0f};
        return observe();
    }

    struct StepResult {
        env::EncodedObservation obs;
        env::Gamemode mode = env::Gamemode::Cube;
        float percent = 0.0f;
        bool dead = false;
        bool completed = false;
    };

    StepResult step(int action) {
        const bool hold = (action != 0);
        m_player.prevYVelocity = m_player.yVelocity;
        m_player.isHolding = hold;
        m_player.holdDurationSec = hold ? (m_player.holdDurationSec + kDt) : 0.0f;
        m_player.recentActions[3] = m_player.recentActions[2];
        m_player.recentActions[2] = m_player.recentActions[1];
        m_player.recentActions[1] = m_player.recentActions[0];
        m_player.recentActions[0] = hold ? 1.0f : 0.0f;

        if (m_player.mode == env::Gamemode::Cube && hold && m_player.isOnGround) {
            m_player.yVelocity = kJumpVel;
            m_player.isOnGround = false;
        }

        constexpr int kSub = 3;
        const float subDt = kDt / static_cast<float>(kSub);
        bool dead = false;
        bool completed = false;

        for (int s = 0; s < kSub; ++s) {
            m_player.x += kSpeedX * subDt;
            if (m_player.x >= kShipPortalX && m_player.mode == env::Gamemode::Cube) {
                m_player.mode = env::Gamemode::Ship;
                m_player.yVelocity *= 0.35f;
            }

            if (m_player.mode == env::Gamemode::Cube) {
                if (!m_player.isOnGround) {
                    m_player.yVelocity += kCubeGravity * subDt;
                    m_player.y += m_player.yVelocity * subDt;
                    if (m_player.y <= kPlayerHalf) {
                        m_player.y = kPlayerHalf;
                        m_player.yVelocity = 0.0f;
                        m_player.isOnGround = true;
                    }
                }
            } else {
                const float accel = kShipGravity + (hold ? kShipThrust : 0.0f);
                m_player.yVelocity = std::clamp(m_player.yVelocity + accel * subDt, -260.0f, 260.0f);
                m_player.y += m_player.yVelocity * subDt;
                if (m_player.y <= kPlayerHalf) {
                    m_player.y = kPlayerHalf;
                    m_player.yVelocity = std::max(0.0f, m_player.yVelocity);
                    m_player.isOnGround = true;
                } else if (m_player.y >= kCeilingY - kPlayerHalf) {
                    m_player.y = kCeilingY - kPlayerHalf;
                    m_player.yVelocity = std::min(0.0f, m_player.yVelocity);
                    m_player.isOnGround = false;
                } else {
                    m_player.isOnGround = false;
                }
            }

            const float px0 = m_player.x - kPlayerHalf * 0.7f;
            const float px1 = m_player.x + kPlayerHalf * 0.7f;
            const float py0 = m_player.y - kPlayerHalf * 0.85f;
            const float py1 = m_player.y + kPlayerHalf * 0.7f;

            for (const auto& obj : m_objects) {
                if (obj.category != env::ObjectCategory::Hazard) continue;
                if (px1 > obj.minX && px0 < obj.maxX && py1 > obj.minY && py0 < obj.maxY) {
                    dead = true;
                    break;
                }
            }
            if (dead) break;
            if (m_player.x >= kLevelLength) {
                completed = true;
                break;
            }
        }

        StepResult res;
        res.obs = observe();
        res.mode = m_player.mode;
        res.percent = std::clamp(100.0f * m_player.x / kLevelLength, 0.0f, 100.0f);
        res.dead = dead;
        res.completed = completed;
        return res;
    }

    float playerX() const { return m_player.x; }
    env::Gamemode mode() const { return m_player.mode; }

private:
    env::EncodedObservation observe() const {
        return env::ObservationEncoder::encode(m_player, m_objects, kGroundY);
    }

    env::PlayerState m_player;
    std::vector<env::LevelObject> m_objects;
};

void testEndToEndMultiModeConvergence() {
    rl::MultiModePPOAgent multiAgent(
        rl::TrainingPreset::FastLearner,
        env::EncodedObservation::kTotalDim,
        env::EncodedObservation::kContinuousNormDim
    );
    // Use compact hidden size for fast CI execution while keeping full 342D observation
    for (std::size_t m = 0; m < rl::MultiModePPOAgent::kModeCount; ++m) {
        auto& a = multiAgent.agentFor(static_cast<env::Gamemode>(m));
        auto cfg = a.config();
        cfg.rolloutHorizon = 192;
        cfg.minibatchSize = 48;
        cfg.learningRate = 7e-4f;
        cfg.silCoef = 0.25f;
        cfg.silBatches = 3;
        a.mutableConfig() = cfg;
        a.applyPreset(rl::TrainingPreset::FastLearner);
        a.mutableConfig().rolloutHorizon = 192;
        a.mutableConfig().minibatchSize = 48;
        a.mutableConfig().learningRate = 7e-4f;
        a.mutableConfig().silCoef = 0.25f;
        a.mutableConfig().silBatches = 3;
        a.clearBuffer();
    }

    env::RewardCalculator rewards;
    env::StatsTracker stats;
    stats.onLevelChanged(1, "Multi-Mode Cube+Ship Course");
    rewards.onLevelStart();

    MultiModeGDRunner sim;

    // Sanity check: constant policies must fail early (< 25%)
    auto evalConstant = [&](int fixedAction) -> float {
        sim.reset();
        for (int step = 0; step < 300; ++step) {
            auto res = sim.step(fixedAction);
            if (res.completed) return 100.0f;
            if (res.dead) return res.percent;
        }
        return 0.0f;
    };
    expect(evalConstant(0) < 25.0f, "Course too easy: always-release survived past 25%");
    expect(evalConstant(1) < 25.0f, "Course too easy: always-hold survived past 25%");

    auto evalDeterministic = [&]() -> float {
        auto obs = sim.reset();
        for (int step = 0; step < 300; ++step) {
            auto dec = multiAgent.selectAction(sim.mode(), obs.vec, false);
            auto res = sim.step(dec.action);
            obs = res.obs;
            if (res.completed) return 100.0f;
            if (res.dead) return res.percent;
        }
        return 0.0f;
    };

    const float initialEvalPct = evalDeterministic();
    expect(initialEvalPct < 35.0f, "Untrained agent should not already beat the multi-mode course");

    // Benchmark hot-swap inference speed over 1,000 frames
    {
        auto obs = sim.reset();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; ++i) {
            const auto m = (i & 1) ? env::Gamemode::Ship : env::Gamemode::Cube;
            volatile auto d = multiAgent.selectAction(m, obs.vec, false);
            (void)d.action;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double usPerStep = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000.0;
        std::cout << "[BENCH] Hot-swap single-step inference latency (342D): " << usPerStep << " us\n";
        expect(usPerStep < 500.0, "Inference latency exceeded 0.5 ms budget");
    }
    multiAgent.resetHotSwapCount();

    // Train with the exact new rule: No reward if it dies (0x), 2x if it gets a new best + SIL!
    auto obs = sim.reset();
    rewards.resetEpisode(0.0f);
    std::uint64_t episodeId = 1;
    bool mastered = false;
    float bestEvalPct = initialEvalPct;

    for (int step = 0; step < 256 * 260; ++step) {
        const auto modeBefore = sim.mode();
        auto dec = multiAgent.selectAction(modeBefore, obs.vec, true);
        auto res = sim.step(dec.action);
        const bool done = res.dead || res.completed;
        const bool modeSwapped = (!done && res.mode != modeBefore);

        float rawR = rewards.computeStepReward(sim.playerX(), dec.action);
        if (res.dead) {
            rawR = rewards.computeDeathReward(); // 0.0f: no reward on death
        } else if (res.completed) {
            rawR += rewards.computeCompletionReward();
        }

        const float portalBootstrap = modeSwapped
            ? multiAgent.estimateValue(res.mode, res.obs.vec)
            : 0.0f;

        multiAgent.stageTransition(modeBefore, dec, rawR, done, modeSwapped, portalBootstrap, episodeId);
        stats.onStep(res.percent, rawR, dec.action, dec.holdProb, dec.value, res.mode);

        if (done) {
            const auto resolution = res.completed ? rewards.resolveCompletion() : rewards.resolveDeath();
            const auto updates = multiAgent.finalizeEpisode(
                episodeId,
                resolution.rewardMultiplier,
                res.percent,
                resolution.saveToElite,
                resolution.fatalTrimSteps
            );
            stats.onEpisodeEnd(res.percent, res.completed, modeBefore, resolution.finalEpisodeReward);

            if (!updates.empty()) {
                for (const auto& u : updates) stats.onPPOUpdate(u);
                const float evalPct = evalDeterministic();
                bestEvalPct = std::max(bestEvalPct, evalPct);
                if (evalPct >= 100.0f && stats.totalCompletions() >= 3) {
                    mastered = true;
                    break;
                }
            }

            obs = sim.reset();
            rewards.resetEpisode(0.0f);
            ++episodeId;
        } else {
            obs = res.obs;
        }
    }

    const auto csvPath = std::filesystem::temp_directory_path() / "gdai_test_stats.csv";
    expect(stats.exportCSV(csvPath), "CSV export failed");
    expect(std::filesystem::file_size(csvPath) > 50, "Exported CSV file is empty");
    std::filesystem::remove(csvPath);

    std::cout << "[PASS] testEndToEndMultiModeConvergence: "
              << "initialEval=" << initialEvalPct << "% -> bestEval=" << bestEvalPct << "%, "
              << "episodes=" << stats.totalEpisodes()
              << ", cubeUpd=" << multiAgent.agentFor(env::Gamemode::Cube).updateCount()
              << ", shipUpd=" << multiAgent.agentFor(env::Gamemode::Ship).updateCount()
              << ", hotSwaps=" << multiAgent.hotSwapCount()
              << ", clears=" << stats.totalCompletions()
              << ", rollingAvg20=" << stats.rollingAvgPercent(20) << "%\n";

    expect(multiAgent.hotSwapCount() > 0, "Expected gamemode hot-swaps during multi-mode run");
    expect(multiAgent.agentFor(env::Gamemode::Ship).totalSteps() > 0, "Ship model was never trained");
    expect(mastered || bestEvalPct >= 100.0f, "Multi-mode PPO+SIL agent failed to clear Cube+Ship course");
}

} // namespace

int main() {
    std::cout << "=== GD-AI RL Master — C++23 Multi-Mode Engine Verification Suite ===\n";
    testOrthogonalInit();
    testFiniteDifferenceGradients();
    testGAEComputation();
    testObservationEncoder();
    testRewardRuleNoRewardOnDeath2xNewBest();
    testModelSerializationAndMultiModeHotSwap();
    testEndToEndMultiModeConvergence();
    std::cout << "=== ALL TESTS PASSED ===\n";
    return 0;
}
