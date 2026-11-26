#include "game/AIManager.hpp"

#include <Geode/Geode.hpp>

using namespace geode::prelude;

$on_mod(Loaded) {
    auto& mgr = gdai::game::AIManager::get();
    mgr.initFromSettings();

    listenForSettingChanges<bool>("ai-enabled", [](bool) {
        gdai::game::AIManager::get().syncFromSettings();
    });
    listenForSettingChanges<bool>("training-mode", [](bool) {
        gdai::game::AIManager::get().syncFromSettings();
    });
    listenForSettingChanges<bool>("show-hud", [](bool) {
        gdai::game::AIManager::get().syncFromSettings();
    });
    listenForSettingChanges<bool>("show-vision", [](bool) {
        gdai::game::AIManager::get().syncFromSettings();
    });
    listenForSettingChanges<bool>("per-level-model", [](bool) {
        gdai::game::AIManager::get().syncFromSettings();
    });
    listenForSettingChanges<bool>("auto-save", [](bool) {
        gdai::game::AIManager::get().syncFromSettings();
    });
    listenForSettingChanges<double>("training-speed", [](double) {
        gdai::game::AIManager::get().syncFromSettings();
    });
    listenForSettingChanges<double>("learning-rate", [](double) {
        gdai::game::AIManager::get().syncFromSettings();
    });
    listenForSettingChanges<int64_t>("decision-interval", [](int64_t) {
        gdai::game::AIManager::get().syncFromSettings();
    });

    log::info(
        "[GD-AI RL Master] Loaded! 8x Gamemode Hot-Swap PPO+SIL ({} params/mode, {}D egocentric observation).",
        mgr.agent().network().parameterCount(),
        mgr.agent().config().obsDim
    );
}
