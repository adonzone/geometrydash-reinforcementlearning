# GD-AI RL Master

An in-game **Proximal Policy Optimization (PPO)** reinforcement learning mod for Geometry Dash built with **Geode SDK v6**.

## Features
- **Pure C++23 In-Process PPO Engine**: Zero external Python/PyTorch dependencies—runs directly inside Geometry Dash at sub-millisecond inference latency.
- **240D Egocentric Perception**: Combines player kinematics, gamemode one-hot state (Cube, Ship, Ball, UFO, Wave, Robot, Spider, Swing), precision vectors to nearest hazards/solids/orbs, and a 10×7×3 semantic obstacle grid that generalizes across levels.
- **Live In-Game HUD & Sparkline Chart**: Track Episode #, Best %, Rolling Avg %, Clears, Win Rate, Policy Confidence (`P(HOLD)` bar), Value Estimate `V(s)`, PPO Losses, Entropy, and KL divergence in real time.
- **Real-Time Vision Overlay**: Watch the AI's semantic perception grid and danger rays rendered right onto the level.
- **Full Analytics Dashboard**: Accessible from the HUD (`UI`), Pause menu (`AI`), or Main Menu (`GD-AI`) with training presets, training speed multiplier (`1x`–`4x`), checkpoint save/load, and CSV export.
