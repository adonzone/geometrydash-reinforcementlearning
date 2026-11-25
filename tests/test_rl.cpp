#include "env/Observation.hpp"
#include "env/RewardCalculator.hpp"
#include "env/StatsTracker.hpp"
#include "rl/PPOAgent.hpp"

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

    // Columns should be orthogonal with norm 1.5
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

    // Loss L = 0.7 * logit0 - 0.4 * logit1 + 0.5 * (value - 1.25)^2
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
    rl::RolloutBuffer buf(4);
    // 2-step episode ending at t=1 (done=true), then 2-step non-terminal episode
    buf.add({{}, 0, 0.0f, 1.0f, 2.0f, false});
    buf.add({{}, 1, 0.0f, 1.5f, 3.0f, true});
    buf.add({{}, 0, 0.0f, 0.5f, 1.0f, false});
    buf.add({{}, 1, 0.0f, 0.8f, 1.0f, false});

    const float gamma = 0.9f;
    const float lam = 0.8f;
    const float bootstrap = 2.0f;
    buf.computeGAE(bootstrap, gamma, lam);

    // t=1 (done): delta1 = 3.0 - 1.5 = 1.5 -> A1 = 1.5, R1 = 3.0
    expect(std::abs(buf.advantage(1) - 1.5f) < 1e-5f, "GAE terminal step mismatch");
    expect(std::abs(buf.ret(1) - 3.0f) < 1e-5f, "GAE terminal return mismatch");

    // t=0: delta0 = 2.0 + 0.9*1.5 - 1.0 = 2.35; A0 = 2.35 + 0.9*0.8*1.5 = 3.43
    expect(std::abs(buf.advantage(0) - 3.43f) < 1e-4f, "GAE non-terminal step mismatch");

    // t=3: delta3 = 1.0 + 0.9*2.0 - 0.8 = 2.0 -> A3 = 2.0
    expect(std::abs(buf.advantage(3) - 2.0f) < 1e-5f, "GAE bootstrap step mismatch");
    std::cout << "[PASS] testGAEComputation\n";
}

void testObservationEncoder() {
    env::PlayerState p;
    p.x = 100.0f;
    p.y = 105.0f;
    p.yVelocity = 6.0f;
    p.mode = env::Gamemode::Ship;
    p.isOnGround = false;

    std::vector<env::LevelObject> objs = {
        // Hazard 90 units ahead, same Y
        {175.0f, 90.0f, 205.0f, 120.0f, env::ObjectCategory::Hazard},
        // Solid block 60 units ahead, below player
        {145.0f, 60.0f, 175.0f, 90.0f, env::ObjectCategory::Solid},
        // Orb 120 units ahead, above player
        {205.0f, 135.0f, 235.0f, 165.0f, env::ObjectCategory::Interactive},
    };

    const auto enc = env::ObservationEncoder::encode(p, objs, 0.0f);
    expect(enc.vec.size() == env::EncodedObservation::kTotalDim, "Unexpected observation vector dimension");
    expect(enc.targets.hasHazard1 && std::abs(enc.targets.hazard1Dx - 90.0f) < 1e-3f, "Nearest hazard dx mismatch");
    expect(enc.targets.hasInteractive && std::abs(enc.targets.interactiveDx - 120.0f) < 1e-3f, "Nearest interactive dx mismatch");

    // Ship one-hot is at index 12 + 1 = 13
    expect(enc.vec[13] == 1.0f && enc.vec[12] == 0.0f, "Ship gamemode one-hot mismatch");

    // Hazard cell should be populated in channel 0
    bool foundHazardCell = false;
    for (std::size_t r = 0; r < env::VisionGrid::kRows; ++r) {
        for (std::size_t c = 0; c < env::VisionGrid::kCols; ++c) {
            if (enc.grid.at(c, r, 0) > 0.5f) foundHazardCell = true;
        }
    }
    expect(foundHazardCell, "Hazard not rasterized into vision grid");
    std::cout << "[PASS] testObservationEncoder\n";
}

void testModelSerializationAndChecksum() {
    const auto tmpPath = std::filesystem::temp_directory_path() / "gdai_test_checkpoint.gdai";

    rl::PPOConfig cfg;
    cfg.obsDim = 16;
    cfg.hidden1 = 32;
    cfg.hidden2 = 16;
    cfg.seed = 99;

    rl::PPOAgent agentA(cfg);
    rl::Vec sampleObs(16, 0.25f);
    sampleObs[3] = -0.8f;
    // Update obs norm stats slightly
    const auto dA = agentA.selectAction(sampleObs, true);
    (void)dA;
    const auto evalBefore = agentA.selectAction(sampleObs, false);

    expect(agentA.saveModel(tmpPath), "Failed to save model");

    rl::PPOConfig cfgB = cfg;
    cfgB.seed = 777;
    rl::PPOAgent agentB(cfgB);
    expect(agentB.loadModel(tmpPath), "Failed to load valid model");

    const auto evalAfter = agentB.selectAction(sampleObs, false);
    expect(std::abs(evalBefore.holdProb - evalAfter.holdProb) < 1e-6f, "Reloaded holdProb mismatch");
    expect(std::abs(evalBefore.value - evalAfter.value) < 1e-6f, "Reloaded value mismatch");

    // Flip a byte in the middle of the file and verify checksum catches corruption
    {
        std::fstream fs(tmpPath, std::ios::in | std::ios::out | std::ios::binary);
        fs.seekg(32);
        char b = 0;
        fs.read(&b, 1);
        b ^= 0x55;
        fs.seekp(32);
        fs.write(&b, 1);
    }
    expect(!agentB.loadModel(tmpPath), "Corrupted model should fail checksum verification");
    std::filesystem::remove(tmpPath);
    std::cout << "[PASS] testModelSerializationAndChecksum\n";
}

/// Minimal Geometry Dash Cube simulator to prove end-to-end PPO learning convergence!
class MiniGDRunner {
public:
    static constexpr float kLevelLength = 2100.0f;
    static constexpr float kDt = 1.0f / 30.0f; // matches 60 FPS with decisionInterval=2
    static constexpr float kSpeedX = 310.0f;
    static constexpr float kJumpVel = 520.0f;
    static constexpr float kGravity = -2350.0f;
    static constexpr float kGroundY = 0.0f;
    static constexpr float kPlayerHalf = 14.0f;

    MiniGDRunner() {
        // Uneven floor spikes + overhead ceiling hazards so neither always-HOLD nor always-RELEASE can survive!
        const float floorSpikeXs[] = {310.f, 740.f, 1180.f, 1620.f};
        for (float sx : floorSpikeXs) {
            m_objects.push_back({sx - 12.f, 0.f, sx + 12.f, 26.f, env::ObjectCategory::Hazard});
        }
        // Overhead hazards positioned where mindless auto-jumping collides
        const float ceilSpikeXs[] = {170.f, 520.f, 960.f, 1400.f, 1840.f};
        for (float cx : ceilSpikeXs) {
            m_objects.push_back({cx - 18.f, 44.f, cx + 18.f, 84.f, env::ObjectCategory::Hazard});
        }
        reset();
    }

    env::EncodedObservation reset() {
        m_player = env::PlayerState{};
        m_player.x = 0.0f;
        m_player.y = kPlayerHalf;
        m_player.yVelocity = 0.0f;
        m_player.isOnGround = true;
        m_player.mode = env::Gamemode::Cube;
        return observe();
    }

    struct StepResult {
        env::EncodedObservation obs;
        float percent = 0.0f;
        bool dead = false;
        bool completed = false;
    };

    StepResult step(int action) {
        const bool hold = (action != 0);
        if (hold && m_player.isOnGround) {
            m_player.yVelocity = kJumpVel;
            m_player.isOnGround = false;
        }
        m_player.isHolding = hold;
        m_player.holdDurationSec = hold ? (m_player.holdDurationSec + kDt) : 0.0f;

        // Sub-step physics for crisp collision
        constexpr int kSub = 3;
        const float subDt = kDt / static_cast<float>(kSub);
        bool dead = false;
        bool completed = false;

        for (int s = 0; s < kSub; ++s) {
            m_player.x += kSpeedX * subDt;
            if (!m_player.isOnGround) {
                m_player.yVelocity += kGravity * subDt;
                m_player.y += m_player.yVelocity * subDt;
                if (m_player.y <= kPlayerHalf) {
                    m_player.y = kPlayerHalf;
                    m_player.yVelocity = 0.0f;
                    m_player.isOnGround = true;
                }
            }

            // Check spike collisions
            const float px0 = m_player.x - kPlayerHalf * 0.7f;
            const float px1 = m_player.x + kPlayerHalf * 0.7f;
            const float py0 = m_player.y - kPlayerHalf;
            const float py1 = m_player.y + kPlayerHalf * 0.7f;

            for (const auto& obj : m_objects) {
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
        res.percent = std::clamp(100.0f * m_player.x / kLevelLength, 0.0f, 100.0f);
        res.dead = dead;
        res.completed = completed;
        return res;
    }

    float playerX() const { return m_player.x; }

private:
    env::EncodedObservation observe() const {
        return env::ObservationEncoder::encode(m_player, m_objects, kGroundY);
    }

    env::PlayerState m_player;
    std::vector<env::LevelObject> m_objects;
};

void testEndToEndPPOTrainingConvergence() {
    auto cfg = rl::PPOConfig::fromPreset(rl::TrainingPreset::FastLearner, env::EncodedObservation::kTotalDim);
    cfg.hidden1 = 64;
    cfg.hidden2 = 32;
    cfg.rolloutHorizon = 256;
    cfg.minibatchSize = 64;
    cfg.epochs = 4;
    cfg.learningRate = 8e-4f;
    cfg.seed = 2026;

    rl::PPOAgent agent(cfg);
    env::RewardCalculator rewards;
    env::StatsTracker stats;
    stats.onLevelChanged(1, "Simulated Spike Course");
    rewards.onLevelStart();

    MiniGDRunner sim;

    // Sanity check: trivial constant policies must fail early (< 25%)
    auto evalConstant = [&](int fixedAction) -> float {
        sim.reset();
        for (int step = 0; step < 300; ++step) {
            auto res = sim.step(fixedAction);
            if (res.completed) return 100.0f;
            if (res.dead) return res.percent;
        }
        return 0.0f;
    };
    const float alwaysReleasePct = evalConstant(0);
    const float alwaysHoldPct = evalConstant(1);
    expect(alwaysReleasePct < 25.0f, "Course too easy: always-release survived past 25%");
    expect(alwaysHoldPct < 25.0f, "Course too easy: always-hold survived past 25%");

    // Measure baseline untrained deterministic performance
    auto evalDeterministic = [&]() -> float {
        auto obs = sim.reset();
        for (int step = 0; step < 300; ++step) {
            auto dec = agent.selectAction(obs.vec, false);
            auto res = sim.step(dec.action);
            obs = res.obs;
            if (res.completed) return 100.0f;
            if (res.dead) return res.percent;
        }
        return 0.0f;
    };

    const float initialEvalPct = evalDeterministic();
    expect(initialEvalPct < 35.0f, "Untrained agent should not already beat the course");

    // Benchmark inference speed over 1,000 frames
    {
        auto obs = sim.reset();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; ++i) {
            volatile auto d = agent.selectAction(obs.vec, false);
            (void)d.action;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double usPerStep = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000.0;
        std::cout << "[BENCH] Single-step inference latency: " << usPerStep << " us\n";
        expect(usPerStep < 500.0, "Inference latency exceeded 0.5 ms budget");
    }

    // Train via PPO until the agent consistently beats the course (or max 120 updates)
    auto obs = sim.reset();
    rewards.resetEpisode(0.0f);
    bool mastered = false;
    float bestEvalPct = initialEvalPct;

    for (int step = 0; step < 256 * 120; ++step) {
        auto dec = agent.selectAction(obs.vec, true);
        auto res = sim.step(dec.action);

        float r = rewards.computeStepReward(sim.playerX(), dec.action);
        if (res.dead) r += rewards.computeDeathReward();
        if (res.completed) r += rewards.computeCompletionReward();

        stats.onStep(res.percent, r, dec.action, dec.holdProb, dec.value, env::Gamemode::Cube);
        const bool done = res.dead || res.completed;
        auto upd = agent.recordTransition(dec, r, done, done ? nullptr : &res.obs.vec);
        if (upd) {
            stats.onPPOUpdate(*upd);
            const float evalPct = evalDeterministic();
            bestEvalPct = std::max(bestEvalPct, evalPct);
            if (evalPct >= 100.0f && stats.totalCompletions() >= 3) {
                mastered = true;
                break;
            }
            // Re-sync sim after deterministic eval probe
            obs = sim.reset();
            rewards.resetEpisode(0.0f);
            continue;
        }

        if (done) {
            stats.onEpisodeEnd(res.percent, res.completed, env::Gamemode::Cube);
            obs = sim.reset();
            rewards.resetEpisode(0.0f);
        } else {
            obs = res.obs;
        }
    }

    // Verify CSV export works
    const auto csvPath = std::filesystem::temp_directory_path() / "gdai_test_stats.csv";
    expect(stats.exportCSV(csvPath), "CSV export failed");
    expect(std::filesystem::file_size(csvPath) > 50, "Exported CSV file is empty");
    std::filesystem::remove(csvPath);

    std::cout << "[PASS] testEndToEndPPOTrainingConvergence: "
              << "initialEval=" << initialEvalPct << "% -> bestEval=" << bestEvalPct << "%, "
              << "episodes=" << stats.totalEpisodes()
              << ", updates=" << stats.totalUpdates()
              << ", clears=" << stats.totalCompletions()
              << ", rollingAvg20=" << stats.rollingAvgPercent(20) << "%\n";

    expect(mastered || bestEvalPct >= 100.0f, "PPO agent failed to clear the simulated Geometry Dash course");
}

} // namespace

int main() {
    std::cout << "=== GD-AI RL Master — C++23 Engine Verification Suite ===\n";
    testOrthogonalInit();
    testFiniteDifferenceGradients();
    testGAEComputation();
    testObservationEncoder();
    testModelSerializationAndChecksum();
    testEndToEndPPOTrainingConvergence();
    std::cout << "=== ALL TESTS PASSED ===\n";
    return 0;
}
