#include "StatsOverlay.hpp"
#include "AIDashboardPopup.hpp"
#include "../game/AIManager.hpp"

using namespace geode::prelude;

namespace gdai::ui {

StatsOverlay* StatsOverlay::create() {
    auto* ret = new StatsOverlay();
    if (ret && ret->init()) {
        ret->autorelease();
        return ret;
    }
    CC_SAFE_DELETE(ret);
    return nullptr;
}

bool StatsOverlay::init() {
    if (!CCNode::init()) return false;

    const auto winSize = CCDirector::sharedDirector()->getWinSize();
    constexpr float kW = 192.0f;
    constexpr float kH = 152.0f;

    this->setContentSize({kW, kH});
    this->setPosition({winSize.width - kW - 8.0f, winSize.height - kH - 8.0f});

    m_bg = CCLayerColor::create({12, 16, 24, 196}, kW, kH);
    this->addChild(m_bg, 0);

    // Subtle border
    auto* border = CCDrawNode::create();
    CCPoint pts[4] = {{0.f, 0.f}, {kW, 0.f}, {kW, kH}, {0.f, kH}};
    border->drawPolygon(pts, 4, ccc4f(0, 0, 0, 0), 0.8f, ccc4f(0.32f, 0.68f, 0.96f, 0.55f));
    this->addChild(border, 1);

    auto makeLabel = [&](const char* text, float scale, CCPoint pos, ccColor3B color) {
        auto* lbl = CCLabelBMFont::create(text, "chatFont.fnt");
        lbl->setAnchorPoint({0.f, 1.f});
        lbl->setScale(scale);
        lbl->setPosition(pos);
        lbl->setColor(color);
        this->addChild(lbl, 2);
        return lbl;
    };

    m_headerLabel = makeLabel("GD-AI PPO", 0.58f, {7.f, kH - 5.f}, {130, 220, 255});
    m_statusBadge = makeLabel("TRAIN 1.0x", 0.52f, {70.f, kH - 6.f}, {95, 250, 145});

    // Quick Dashboard button in top-right of HUD
    auto* dashSpr = ButtonSprite::create("UI", "bigFont.fnt", "GJ_button_05.png", 0.35f);
    dashSpr->setScale(0.45f);
    auto* dashBtn = CCMenuItemExt::createSpriteExtra(
        dashSpr,
        [](CCObject*) {
            AIDashboardPopup::create()->show();
        }
    );
    auto* menu = CCMenu::create();
    menu->setPosition({kW - 18.f, kH - 11.f});
    menu->addChild(dashBtn);
    this->addChild(menu, 3);

    m_line1 = makeLabel("Ep: 0 | Upd: 0 | Clears: 0", 0.48f, {7.f, kH - 22.f}, {235, 240, 248});
    m_line2 = makeLabel("Best: 0.0% | Avg20: 0.0%", 0.48f, {7.f, kH - 34.f}, {255, 220, 105});
    m_line3 = makeLabel("Rew: +0.0 | V(s): +0.0", 0.48f, {7.f, kH - 46.f}, {180, 235, 200});
    m_line4 = makeLabel("P:0.000 V:0.000 Ent:0.69", 0.46f, {7.f, kH - 58.f}, {175, 195, 225});

    // Live Policy P(HOLD) gauge
    m_actionLabel = makeLabel("P(HOLD): 50% [REL]", 0.46f, {7.f, kH - 72.f}, {225, 235, 250});
    m_holdGauge = ChartNode::create(kW - 14.f, 8.f);
    m_holdGauge->setPosition({7.f, kH - 92.f});
    this->addChild(m_holdGauge, 2);

    // Live episode progress chart (% + rolling mean)
    m_progressChart = ChartNode::create(kW - 14.f, 48.f);
    m_progressChart->setPosition({7.f, 7.f});
    m_progressChart->setFixedRange(0.0f, 100.0f);
    this->addChild(m_progressChart, 2);

    refresh();
    return true;
}

void StatsOverlay::refresh() {
    const auto& mgr = game::AIManager::get();
    this->setVisible(mgr.showHUD());
    if (!mgr.showHUD()) return;

    const auto& st = mgr.stats();

    // Update live action state, P(HOLD) bar & active hot-swapped gamemode brain every frame
    const float holdP = st.lastHoldProb();
    const int act = st.lastAction();
    const char* actStr = (act == 2) ? "HOLD" : (act == 1 ? "CLICK" : "REL");
    m_actionLabel->setString(fmt::format(
        "Act: [{}] P(H):{:2.0f}% | Brain:{}",
        actStr,
        holdP * 100.0f,
        env::gamemodeName(mgr.activeGamemode())
    ).c_str());
    m_actionLabel->setColor(
        (act == 2) ? ccColor3B{110, 255, 150} :
        ((act == 1) ? ccColor3B{255, 225, 90} : ccColor3B{195, 210, 230})
    );
    m_holdGauge->renderGauge(
        holdP,
        (act == 2) ? ccc4f(0.22f, 0.92f, 0.48f, 0.92f) :
        ((act == 1) ? ccc4f(0.98f, 0.85f, 0.25f, 0.92f) : ccc4f(0.32f, 0.62f, 0.96f, 0.85f))
    );

    // Throttle text & sparkline chart updates slightly to minimize overhead
    if (++m_frameSkip < 4) return;
    m_frameSkip = 0;

    if (!mgr.isAIEnabled()) {
        m_statusBadge->setString("OFF");
        m_statusBadge->setColor({235, 120, 120});
    } else if (mgr.isTrainingMode()) {
        m_statusBadge->setString(fmt::format(
            "TRAIN {:.1f}x{}",
            mgr.trainingSpeed(),
            mgr.brokeRecordThisEpisode() ? " 2xPB!" : ""
        ).c_str());
        m_statusBadge->setColor(mgr.brokeRecordThisEpisode() ? ccColor3B{255, 225, 90} : ccColor3B{95, 250, 145});
    } else {
        m_statusBadge->setString("EVAL PLAY");
        m_statusBadge->setColor({95, 215, 255});
    }

    m_line1->setString(fmt::format(
        "Ep:{} | Upd:{} | Win:{} | Swp:{}",
        st.totalEpisodes(),
        mgr.multiAgent().totalUpdatesAll(),
        st.totalCompletions(),
        mgr.multiAgent().hotSwapCount()
    ).c_str());

    m_line2->setString(fmt::format(
        "Now:{:.1f}% | Best:{:.1f}% | Avg:{:.1f}%",
        st.currentPercent(),
        st.bestPercent(),
        st.rollingAvgPercent(20)
    ).c_str());

    m_line3->setString(fmt::format(
        "LastR:{:+.1f} | AvgR:{:+.1f} | V:{:+.1f}",
        st.lastEpisodeReward(),
        st.rollingAvgReward(20),
        st.lastValueEstimate()
    ).c_str());

    const auto& activeAgent = mgr.agent();
    const auto& u = activeAgent.lastMetrics();
    m_line4->setString(fmt::format(
        "P:{:.3f} Ent:{:.2f} SIL:{} Buf:{}/{}",
        u.policyLoss,
        u.entropy,
        activeAgent.eliteCount(),
        activeAgent.bufferFill(),
        activeAgent.bufferCapacity()
    ).c_str());

    ChartSeries rawPct;
    rawPct.values = st.percentSeries();
    rawPct.color = ccc4f(0.28f, 0.88f, 0.60f, 0.65f);
    rawPct.lineThickness = 0.9f;

    ChartSeries avgPct;
    avgPct.values = st.rollingPercentSeries(10);
    avgPct.color = ccc4f(1.0f, 0.82f, 0.22f, 0.98f);
    avgPct.lineThickness = 1.4f;

    m_progressChart->renderSeries({rawPct, avgPct});
}

} // namespace gdai::ui
