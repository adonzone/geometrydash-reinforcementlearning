#include "AIDashboardPopup.hpp"
#include "../game/AIManager.hpp"

#include <cmath>
#include <functional>

using namespace geode::prelude;

namespace gdai::ui {

namespace {
constexpr float kPopupW = 450.0f;
constexpr float kPopupH = 295.0f;
} // namespace

AIDashboardPopup* AIDashboardPopup::create() {
    auto* ret = new AIDashboardPopup();
    if (ret && ret->init()) {
        ret->autorelease();
        return ret;
    }
    CC_SAFE_DELETE(ret);
    return nullptr;
}

bool AIDashboardPopup::init() {
    if (!Popup::init(kPopupW, kPopupH)) return false;

    this->setTitle("GD-AI RL Master — PPO Dashboard");

    // Top tab bar
    auto* tabMenu = CCMenu::create();
    tabMenu->setPosition({kPopupW * 0.5f, kPopupH - 48.0f});

    auto makeTabBtn = [&](const char* title, Tab tab, float xOffset) {
        auto* spr = ButtonSprite::create(title, "goldFont.fnt", "GJ_button_04.png", 0.65f);
        spr->setScale(0.72f);
        auto* btn = CCMenuItemExt::createSpriteExtra(
            spr,
            [this, tab](CCObject*) {
                this->switchTab(tab);
            }
        );
        btn->setPosition({xOffset, 0.0f});
        tabMenu->addChild(btn);
    };

    makeTabBtn("Controls", Tab::Controls, -115.0f);
    makeTabBtn("Analytics", Tab::Analytics, 0.0f);
    makeTabBtn("Models & PPO", Tab::Models, 115.0f);
    this->m_mainLayer->addChild(tabMenu, 5);

    m_contentLayer = CCNode::create();
    m_contentLayer->setContentSize({kPopupW - 24.0f, kPopupH - 84.0f});
    m_contentLayer->setPosition({12.0f, 22.0f});
    this->m_mainLayer->addChild(m_contentLayer, 2);

    m_toastLabel = CCLabelBMFont::create("", "chatFont.fnt");
    m_toastLabel->setScale(0.58f);
    m_toastLabel->setPosition({kPopupW * 0.5f, 14.0f});
    m_toastLabel->setColor({120, 255, 165});
    this->m_mainLayer->addChild(m_toastLabel, 6);

    rebuildTabContent();
    return true;
}

void AIDashboardPopup::switchTab(Tab tab) {
    m_activeTab = tab;
    rebuildTabContent();
}

void AIDashboardPopup::rebuildTabContent() {
    if (!m_contentLayer) return;
    m_contentLayer->removeAllChildren();

    switch (m_activeTab) {
        case Tab::Controls:  buildControlsTab(m_contentLayer); break;
        case Tab::Analytics: buildAnalyticsTab(m_contentLayer); break;
        case Tab::Models:    buildModelsTab(m_contentLayer); break;
    }
}

void AIDashboardPopup::buildControlsTab(CCNode* parent) {
    auto& mgr = game::AIManager::get();
    const auto size = parent->getContentSize();

    auto* menu = CCMenu::create();
    menu->setPosition({0.f, 0.f});
    parent->addChild(menu);

    auto addToggle = [&](const std::string& label, bool active, CCPoint pos, std::function<void()> onClick) {
        auto* spr = ButtonSprite::create(
            label.c_str(),
            "bigFont.fnt",
            active ? "GJ_button_01.png" : "GJ_button_06.png",
            0.55f
        );
        spr->setScale(0.58f);
        auto* btn = CCMenuItemExt::createSpriteExtra(
            spr,
            [this, cb = std::move(onClick)](CCObject*) {
                cb();
                this->rebuildTabContent();
            }
        );
        btn->setPosition(pos);
        menu->addChild(btn);
    };

    const float col1 = size.width * 0.27f;
    const float col2 = size.width * 0.73f;

    addToggle(
        mgr.isAIEnabled() ? "AI Agent: ON" : "AI Agent: OFF",
        mgr.isAIEnabled(),
        {col1, size.height - 26.f},
        [&mgr]() { mgr.setAIEnabled(!mgr.isAIEnabled()); }
    );

    addToggle(
        mgr.isTrainingMode() ? "Mode: TRAIN (PPO)" : "Mode: PLAY (EVAL)",
        mgr.isTrainingMode(),
        {col2, size.height - 26.f},
        [&mgr]() { mgr.setTrainingMode(!mgr.isTrainingMode()); }
    );

    addToggle(
        mgr.showHUD() ? "Stats HUD: ON" : "Stats HUD: OFF",
        mgr.showHUD(),
        {col1, size.height - 64.f},
        [&mgr]() { mgr.setShowHUD(!mgr.showHUD()); }
    );

    addToggle(
        mgr.showVision() ? "Vision Grid: ON" : "Vision Grid: OFF",
        mgr.showVision(),
        {col2, size.height - 64.f},
        [&mgr]() { mgr.setShowVision(!mgr.showVision()); }
    );

    addToggle(
        mgr.perLevelModel() ? "Brain: Per-Level" : "Brain: Global Master",
        !mgr.perLevelModel(),
        {col1, size.height - 102.f},
        [&mgr]() { mgr.setPerLevelModel(!mgr.perLevelModel()); }
    );

    // Training Speed selector row
    auto* speedTitle = CCLabelBMFont::create(
        fmt::format("Training Speed: {:.1f}x", mgr.trainingSpeed()).c_str(),
        "goldFont.fnt"
    );
    speedTitle->setScale(0.55f);
    speedTitle->setPosition({col2, size.height - 92.f});
    parent->addChild(speedTitle);

    const float speeds[4] = {1.0f, 1.5f, 2.0f, 4.0f};
    for (int i = 0; i < 4; ++i) {
        const float sp = speeds[i];
        const bool selected = std::abs(mgr.trainingSpeed() - sp) < 0.1f;
        auto* spr = ButtonSprite::create(
            fmt::format("{:.1f}x", sp).c_str(),
            "bigFont.fnt",
            selected ? "GJ_button_01.png" : "GJ_button_05.png",
            0.45f
        );
        spr->setScale(0.45f);
        auto* btn = CCMenuItemExt::createSpriteExtra(
            spr,
            [this, &mgr, sp](CCObject*) {
                mgr.setTrainingSpeed(sp);
                this->rebuildTabContent();
            }
        );
        btn->setPosition({col2 - 60.f + static_cast<float>(i) * 40.f, size.height - 114.f});
        menu->addChild(btn);
    }

    // Live quick summary strip at bottom of Controls tab
    const auto& st = mgr.stats();
    auto* summary = CCLabelBMFont::create(
        fmt::format(
            "Active Level: {}  |  Model: {}\n"
            "Episodes: {}  |  Total Steps: {}  |  Clears: {}  |  Best: {:.1f}%  |  Avg20: {:.1f}%",
            st.levelName(),
            mgr.activeModelLabel(),
            st.totalEpisodes(),
            st.totalSteps(),
            st.totalCompletions(),
            st.bestPercent(),
            st.rollingAvgPercent(20)
        ).c_str(),
        "chatFont.fnt"
    );
    summary->setScale(0.58f);
    summary->setAlignment(kCCTextAlignmentCenter);
    summary->setPosition({size.width * 0.5f, 44.f});
    summary->setColor({210, 232, 255});
    parent->addChild(summary);
}

void AIDashboardPopup::buildAnalyticsTab(CCNode* parent) {
    const auto& mgr = game::AIManager::get();
    const auto& st = mgr.stats();
    const auto size = parent->getContentSize();

    // Metric switcher buttons
    auto* menu = CCMenu::create();
    menu->setPosition({0.f, 0.f});
    parent->addChild(menu);

    auto addMetricBtn = [&](const char* name, ChartMetric m, float x) {
        auto* spr = ButtonSprite::create(
            name,
            "bigFont.fnt",
            m_chartMetric == m ? "GJ_button_01.png" : "GJ_button_05.png",
            0.45f
        );
        spr->setScale(0.46f);
        auto* btn = CCMenuItemExt::createSpriteExtra(
            spr,
            [this, m](CCObject*) {
                m_chartMetric = m;
                this->rebuildTabContent();
            }
        );
        btn->setPosition({x, size.height - 16.f});
        menu->addChild(btn);
    };

    addMetricBtn("Progress %", ChartMetric::Progress, 68.f);
    addMetricBtn("Reward", ChartMetric::Reward, 152.f);
    addMetricBtn("PPO Loss/Ent", ChartMetric::PPODiagnostics, 245.f);

    // Large live chart on the left
    auto* chart = ChartNode::create(262.f, 128.f);
    chart->setPosition({8.f, 24.f});
    parent->addChild(chart);

    std::string legendText;
    if (m_chartMetric == ChartMetric::Progress) {
        chart->setFixedRange(0.0f, 100.0f);
        chart->renderSeries({
            {st.percentSeries(), ccc4f(0.28f, 0.90f, 0.58f, 0.65f), 1.0f},
            {st.rollingPercentSeries(10), ccc4f(1.0f, 0.82f, 0.22f, 1.0f), 1.6f}
        });
        legendText = "Green: Episode %   |   Gold: 10-Ep Rolling Mean %";
    } else if (m_chartMetric == ChartMetric::Reward) {
        chart->setAutoRange();
        chart->renderSeries({
            {st.rewardSeries(), ccc4f(0.32f, 0.78f, 1.0f, 0.95f), 1.4f}
        });
        legendText = "Cyan: Episode Total Shaped Return";
    } else {
        chart->setAutoRange();
        chart->renderSeries({
            {st.entropySeries(), ccc4f(1.0f, 0.82f, 0.22f, 0.95f), 1.4f},
            {st.policyLossSeries(), ccc4f(1.0f, 0.35f, 0.45f, 0.90f), 1.2f}
        });
        legendText = "Gold: Policy Entropy   |   Red: PPO Surrogate Loss";
    }

    auto* legend = CCLabelBMFont::create(legendText.c_str(), "chatFont.fnt");
    legend->setScale(0.50f);
    legend->setPosition({139.f, 12.f});
    legend->setColor({185, 210, 235});
    parent->addChild(legend);

    // KPI table on the right
    const auto& u = st.lastUpdate();
    auto* kpi = CCLabelBMFont::create(
        fmt::format(
            "Best %: {:.1f}%\n"
            "Avg20 %: {:.1f}%\n"
            "Clears: {} ({:.1f}%)\n"
            "Episodes: {}\n"
            "Steps: {}\n"
            "PPO Updates: {}\n"
            "Avg20 Rew: {:+.1f}\n"
            "Exp Var: {:.2f}\n"
            "Approx KL: {:.4f}\n"
            "Clip Frac: {:.1f}%\n"
            "Params: {}k",
            st.bestPercent(),
            st.rollingAvgPercent(20),
            st.totalCompletions(),
            st.rollingWinRate(50),
            st.totalEpisodes(),
            st.totalSteps(),
            st.totalUpdates(),
            st.rollingAvgReward(20),
            u.explainedVariance,
            u.approxKL,
            u.clipFraction * 100.0f,
            mgr.agent().network().parameterCount() / 1000
        ).c_str(),
        "chatFont.fnt"
    );
    kpi->setAnchorPoint({0.f, 1.f});
    kpi->setScale(0.55f);
    kpi->setPosition({280.f, size.height - 10.f});
    kpi->setColor({235, 242, 250});
    parent->addChild(kpi);
}

void AIDashboardPopup::buildModelsTab(CCNode* parent) {
    auto& mgr = game::AIManager::get();
    const auto size = parent->getContentSize();

    auto* menu = CCMenu::create();
    menu->setPosition({0.f, 0.f});
    parent->addChild(menu);

    auto* presetLbl = CCLabelBMFont::create("Training Presets:", "goldFont.fnt");
    presetLbl->setScale(0.56f);
    presetLbl->setPosition({size.width * 0.5f, size.height - 14.f});
    parent->addChild(presetLbl);

    auto addPreset = [&](const char* title, rl::TrainingPreset p, float x) {
        const bool active = mgr.activePreset() == p;
        auto* spr = ButtonSprite::create(
            title,
            "bigFont.fnt",
            active ? "GJ_button_01.png" : "GJ_button_05.png",
            0.50f
        );
        spr->setScale(0.52f);
        auto* btn = CCMenuItemExt::createSpriteExtra(
            spr,
            [this, &mgr, p, title](CCObject*) {
                mgr.applyPreset(p);
                m_toastLabel->setString(fmt::format("Applied preset: {}", title).c_str());
                this->rebuildTabContent();
            }
        );
        btn->setPosition({x, size.height - 42.f});
        menu->addChild(btn);
    };

    addPreset("Fast Learner", rl::TrainingPreset::FastLearner, size.width * 0.20f);
    addPreset("Balanced Master", rl::TrainingPreset::BalancedMaster, size.width * 0.50f);
    addPreset("Fine-Tune", rl::TrainingPreset::FineTunePolish, size.width * 0.80f);

    const auto& cfg = mgr.agent().config();
    auto* cfgSummary = CCLabelBMFont::create(
        fmt::format(
            "Architecture: 240D Egocentric -> Actor(128->64->2) + Critic(128->64->1) [{} params]\n"
            "Horizon: {}  |  Minibatch: {}  |  Epochs: {}  |  LR: {:.1e}  |  Clip: {:.2f}  |  GAE: {:.2f}\n"
            "Checkpoint: {}",
            mgr.agent().network().parameterCount(),
            cfg.rolloutHorizon,
            cfg.minibatchSize,
            cfg.epochs,
            cfg.learningRate,
            cfg.clipEpsilon,
            cfg.gaeLambda,
            mgr.currentModelPath().filename().string()
        ).c_str(),
        "chatFont.fnt"
    );
    cfgSummary->setAlignment(kCCTextAlignmentCenter);
    cfgSummary->setScale(0.52f);
    cfgSummary->setPosition({size.width * 0.5f, size.height - 92.f});
    cfgSummary->setColor({205, 226, 248});
    parent->addChild(cfgSummary);

    // Model & data action buttons
    auto addActionBtn = [&](const char* title, const char* bg, CCPoint pos, std::function<void()> fn) {
        auto* spr = ButtonSprite::create(title, "bigFont.fnt", bg, 0.52f);
        spr->setScale(0.55f);
        auto* btn = CCMenuItemExt::createSpriteExtra(
            spr,
            [cb = std::move(fn)](CCObject*) { cb(); }
        );
        btn->setPosition(pos);
        menu->addChild(btn);
    };

    addActionBtn("Save Model", "GJ_button_01.png", {size.width * 0.20f, 38.f}, [this, &mgr]() {
        const bool ok = mgr.saveCurrentModel();
        m_toastLabel->setString(ok ? "Saved model checkpoint!" : "Failed to save model.");
    });

    addActionBtn("Load Model", "GJ_button_05.png", {size.width * 0.41f, 38.f}, [this, &mgr]() {
        const bool ok = mgr.loadCurrentModel();
        m_toastLabel->setString(ok ? "Loaded saved model checkpoint!" : "No saved checkpoint found yet.");
        this->rebuildTabContent();
    });

    addActionBtn("Export CSV", "GJ_button_04.png", {size.width * 0.62f, 38.f}, [this, &mgr]() {
        const bool ok = mgr.exportStatsCSV();
        m_toastLabel->setString(ok ? "Exported episode metrics CSV to mod save folder!" : "CSV export failed.");
    });

    addActionBtn("Reset Brain", "GJ_button_06.png", {size.width * 0.83f, 38.f}, [this, &mgr]() {
        mgr.resetModelAndStats();
        m_toastLabel->setString("Reset neural weights & training statistics.");
        this->rebuildTabContent();
    });
}

} // namespace gdai::ui
