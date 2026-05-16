# C-UAS Drone Interception with Reinforcement Learning

A physics-based reinforcement learning project that trains a defender drone to autonomously intercept intruder UAVs in a simulated 3D environment. Built from the ground up as a personal deep learning project to understand how RL agents learn spatial pursuit strategies through trial and error.

---

## Overview

This project implements a counter-unmanned aerial systems (C-UAS) simulation where a defender drone learns to track and intercept an intruder UAV using reinforcement learning. The environment is physically simulated using PyBullet, and the agent is trained using Stable-Baselines3 with Gymnasium-compatible custom environments.

The focus was on understanding how RL actually works in continuous 3D space, how reward shaping influences agent behaviour, and what it takes to go from a blank environment to a trained, evaluable policy.

---

## Motivation

Drone interception is a genuinely hard problem. The defender has to learn to predict movement, close distance efficiently, and adjust to an unpredictable target, all without being hand-coded with explicit rules. That is exactly what made it interesting to me.

I wanted to understand how RL agents build spatial intuition from scratch. How does reward shaping change what the agent learns? What happens when you give it too much information versus too little? What does a policy look like before and after convergence?

This project was my attempt to answer those questions by building the whole pipeline myself, from the custom Gymnasium environment to the training loop, evaluation scripts, and visualisation.

---

## Features

- Custom Gymnasium-compatible 3D environment built with PyBullet physics simulation
- Defender drone trained to intercept a moving intruder UAV using model-free RL
- Support for multiple RL algorithms via Stable-Baselines3 (PPO, SAC, TD3)
- Modular project structure separating environments, training, evaluation, detection, and scripts
- TensorBoard and Weights & Biases (wandb) integration for training visualisation and experiment tracking
- Detection module for identifying and tracking intruder position in the simulation
- Evaluation pipeline to measure interception success rate and agent performance across episodes
- Jupyter notebook for interactive experimentation and result analysis

---

## How It Works

1. **Environment:** A custom Gymnasium environment defines the 3D state space, action space, physics simulation (via PyBullet), and reward function. The defender receives the relative position and velocity of the intruder as observations.

2. **Reward shaping:** The agent is rewarded for closing distance to the intruder and penalised for time elapsed and large unnecessary movements. Interception triggers a large positive reward. Getting the reward function right was one of the most iterative parts of this project.

3. **Training:** Stable-Baselines3 handles the RL training loop. The agent explores the environment, collects experience, and updates its policy to maximise cumulative reward over episodes.

4. **Detection:** The detection module processes the simulated environment state to track intruder position, feeding it into the agent's observation at each timestep.

5. **Evaluation:** Trained policies are evaluated across multiple episodes to measure interception rate, time to intercept, and trajectory efficiency.

```
Simulation Environment (PyBullet)
        |
        v
Custom Gymnasium Env (Envs/)
        |
        v
RL Training (Stable-Baselines3 via Training/)
        |
        v
Detection + Observation Pipeline (Detection/)
        |
        v
Policy Evaluation (Evaluation/)
        |
        v
Results + Visualisation (TensorBoard / wandb / Notebook/)
```

---

## Project Structure

```
C-UAS-drone-interception-rl/
│
├── Envs/           # Custom Gymnasium environment definitions
├── Training/       # Training scripts and RL configuration
├── Detection/      # Intruder detection and tracking logic
├── Evaluation/     # Policy evaluation and performance metrics
├── Scripts/        # Utility and helper scripts
├── Notebook/       # Jupyter notebooks for experimentation
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python 3
- PyTorch (deep learning backend)
- Stable-Baselines3 (RL algorithms: PPO, SAC, TD3)
- Gymnasium (custom environment interface)
- PyBullet (physics-based 3D simulation)
- TensorBoard and Weights & Biases (training monitoring)
- OpenCV and Pillow (visual processing)
- NumPy, Pandas, Matplotlib, Seaborn (data processing and visualisation)

---

## How to Run

```bash
# Step 1: Clone the repository
git clone https://github.com/md2801/C-UAS-drone-interception-rl.git
cd C-UAS-drone-interception-rl

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train the agent
python Training/train.py

# Step 4: Evaluate a trained policy
python Evaluation/evaluate.py

# Step 5: (Optional) Launch TensorBoard to monitor training
tensorboard --logdir logs/
```

For interactive exploration, open the notebook in `Notebook/` using Jupyter:

```bash
jupyter notebook Notebook/
```

---

## Learning Outcomes

- Understood how to design a custom Gymnasium environment from scratch, including defining observation spaces, action spaces, step logic, and reset behaviour
- Learned how reward shaping is one of the most consequential design decisions in RL. Small changes to the reward function produced dramatically different agent behaviours
- Gained hands-on experience with model-free RL algorithms and developed intuition for when PPO, SAC, or TD3 are more appropriate
- Discovered how 3D pursuit problems differ from simpler 2D RL tasks, particularly around sparse rewards and exploration in continuous action spaces
- Learned to use TensorBoard and wandb to track episode rewards, policy loss, and value loss across training runs, which made debugging policy behaviour much more tractable
- Understood how physics simulation affects training stability and why curriculum learning or reward annealing can help in environments with complex dynamics

---

## Future Improvements

- Implement a moving, evasive intruder with its own policy for adversarial training
- Add multi-agent support so multiple defenders can coordinate interception
- Introduce wind and drag physics to make the environment more realistic
- Experiment with curriculum learning to progressively increase task difficulty during training
- Add a visual rendering mode with trajectory overlays for clearer policy evaluation
- Explore imitation learning as a warm-start before RL fine-tuning
