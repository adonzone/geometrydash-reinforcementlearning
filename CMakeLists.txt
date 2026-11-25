cmake_minimum_required(VERSION 3.21)

if (GDAI_BUILD_TESTS AND NOT DEFINED ENV{GEODE_SDK})
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if ("${CMAKE_SYSTEM_NAME}" STREQUAL "iOS" OR IOS)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
else()
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
endif()

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

project(GD-AI-RL-Master VERSION 1.0.0 LANGUAGES CXX)

option(GDAI_BUILD_TESTS "Build standalone C++23 RL engine unit & benchmark tests (no Geode SDK required)" OFF)

# Pure C++23 RL core + environment logic (zero Geode / Cocos2d dependencies)
set(GDAI_CORE_SOURCES
    src/rl/PolicyNetwork.cpp
    src/rl/PPOAgent.cpp
    src/env/Observation.cpp
    src/env/StatsTracker.cpp
)

add_library(gdai_core STATIC ${GDAI_CORE_SOURCES})
target_include_directories(gdai_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
if (MSVC)
    target_compile_options(gdai_core PRIVATE /W4 /O2)
else()
    target_compile_options(gdai_core PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

if (GDAI_BUILD_TESTS)
    enable_testing()
    add_executable(gdai_tests tests/test_rl.cpp)
    target_link_libraries(gdai_tests PRIVATE gdai_core)
    add_test(NAME gdai_rl_suite COMMAND gdai_tests)
endif()

# Geode Mod shared library target (built whenever GEODE_SDK is defined or not in test-only mode)
if (DEFINED ENV{GEODE_SDK} OR NOT GDAI_BUILD_TESTS)
    if (NOT DEFINED ENV{GEODE_SDK})
        message(FATAL_ERROR "Unable to find Geode SDK! Define GEODE_SDK or pass -DGDAI_BUILD_TESTS=ON to build standalone RL tests.")
    else()
        message(STATUS "Found Geode SDK: $ENV{GEODE_SDK}")
    endif()

    set(GDAI_MOD_SOURCES
        src/main.cpp
        src/game/AIManager.cpp
        src/game/GameBridge.cpp
        src/game/PlayLayerHooks.cpp
        src/ui/ChartNode.cpp
        src/ui/VisionOverlay.cpp
        src/ui/StatsOverlay.cpp
        src/ui/AIDashboardPopup.cpp
    )

    add_library(${PROJECT_NAME} SHARED ${GDAI_MOD_SOURCES})
    target_link_libraries(${PROJECT_NAME} PRIVATE gdai_core)

    add_subdirectory($ENV{GEODE_SDK} ${CMAKE_CURRENT_BINARY_DIR}/geode)
    setup_geode_mod(${PROJECT_NAME})
endif()
