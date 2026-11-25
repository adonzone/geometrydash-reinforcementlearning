#include "AIManager.hpp"
#include "../ui/AIDashboardPopup.hpp"
#include "../ui/StatsOverlay.hpp"
#include "../ui/VisionOverlay.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/MenuLayer.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

using namespace geode::prelude;

class $modify(GDAIPlayLayer, PlayLayer) {
    struct Fields {
        gdai::ui::StatsOverlay* statsOverlay = nullptr;
        gdai::ui::VisionOverlay* visionOverlay = nullptr;
    };

    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
            return false;
        }

        auto& mgr = gdai::game::AIManager::get();
        mgr.onEnterLevel(this);

        // Attach world-space AI vision grid to the object layer so it tracks level camera
        if (this->m_objectLayer) {
            auto* vision = gdai::ui::VisionOverlay::create();
            vision->setID("gdai-vision-overlay"_spr);
            this->m_objectLayer->addChild(vision, 1200);
            m_fields->visionOverlay = vision;
        }

        // Attach screen-space real-time RL stats HUD to UILayer
        CCNode* hudParent = this->m_uiLayer ? static_cast<CCNode*>(this->m_uiLayer) : static_cast<CCNode*>(this);
        auto* hud = gdai::ui::StatsOverlay::create();
        hud->setID("gdai-stats-hud"_spr);
        hudParent->addChild(hud, 950);
        m_fields->statsOverlay = hud;

        return true;
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);

        auto& mgr = gdai::game::AIManager::get();
        mgr.onPostUpdate(this, dt);

        if (m_fields->visionOverlay) {
            m_fields->visionOverlay->refresh(this);
        }
        if (m_fields->statsOverlay) {
            m_fields->statsOverlay->refresh();
        }
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        if (player == this->m_player1 && !this->m_player1->m_isDead) {
            gdai::game::AIManager::get().onPlayerDeath(this);
        }
        PlayLayer::destroyPlayer(player, object);
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        gdai::game::AIManager::get().onResetLevel(this);
    }

    void levelComplete() {
        gdai::game::AIManager::get().onLevelComplete(this);
        PlayLayer::levelComplete();
    }

    void pauseGame(bool p0) {
        gdai::game::AIManager::get().onPauseChanged(this, true);
        PlayLayer::pauseGame(p0);
    }

    void resume() {
        PlayLayer::resume();
        gdai::game::AIManager::get().onPauseChanged(this, false);
    }

    void onQuit() {
        gdai::game::AIManager::get().onLeaveLevel(this);
        PlayLayer::onQuit();
    }
};

class $modify(GDAIPauseLayer, PauseLayer) {
    void customSetup() {
        PauseLayer::customSetup();

        auto* spr = ButtonSprite::create("AI", "goldFont.fnt", "GJ_button_05.png", 0.75f);
        auto* btn = CCMenuItemExt::createSpriteExtra(
            spr,
            [](CCObject*) {
                gdai::ui::AIDashboardPopup::create()->show();
            }
        );
        btn->setID("gdai-dashboard-btn"_spr);

        if (auto* menu = this->getChildByID("left-button-menu")) {
            menu->addChild(btn);
            menu->updateLayout();
        } else if (auto* rightMenu = this->getChildByID("right-button-menu")) {
            rightMenu->addChild(btn);
            rightMenu->updateLayout();
        } else {
            auto* fallbackMenu = CCMenu::create();
            fallbackMenu->setPosition({45.f, 45.f});
            fallbackMenu->addChild(btn);
            this->addChild(fallbackMenu, 100);
        }
    }
};

class $modify(GDAIMenuLayer, MenuLayer) {
    bool init() {
        if (!MenuLayer::init()) return false;

        auto* spr = ButtonSprite::create("GD-AI", "goldFont.fnt", "GJ_button_01.png", 0.65f);
        auto* btn = CCMenuItemExt::createSpriteExtra(
            spr,
            [](CCObject*) {
                gdai::ui::AIDashboardPopup::create()->show();
            }
        );
        btn->setID("gdai-menu-btn"_spr);

        if (auto* bottomMenu = this->getChildByID("bottom-menu")) {
            bottomMenu->addChild(btn);
            bottomMenu->updateLayout();
        } else {
            auto* menu = CCMenu::create();
            menu->setPosition({55.f, 45.f});
            menu->addChild(btn);
            this->addChild(menu, 50);
        }

        return true;
    }
};
