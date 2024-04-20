# AI Agent for Pacman Game

This repository contains the implementation of a Deep Convolutional Q-Learning Network (DCQN) designed to play the Ms. Pac-Man game using the OpenAI Gymnasium environment. The agent is built using PyTorch and showcases both training and inference phases, including visualization of the gameplay through saved video clips.

# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)

# Installation

To run this project, you will need to install Python along with several dependencies:

```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
pip install gymnasium[box2d]
apt-get install -y swig
pip install torch
pip install torchvision
pip install numpy
pip install Pillow
```

Ensure that your system has CUDA installed if you want to utilize GPU acceleration.

# Usage

To train the agent, simply run the script. By default, the agent will train for 2000 episodes, but this can be adjusted in the code.

```python
python AI_Agent_Pacman.py
```

To visualize the agent's performance after training, you can view the saved video file `video.mp4` that the script generates.

# Code Overview

## Dependencies and Environment Setup
- **Libraries**: The code imports essential libraries like `torch`, `numpy`, `PIL`, and `gymnasium`.
- **Environment**: The Ms. Pac-Man environment (`MsPacmanDeterministic-v0`) is initialized using Gymnasium with a deterministic frame skip setting.

## Preprocessing
- **Frame Processing**: Each frame from the game is resized and converted to a PyTorch tensor to prepare it for input into the convolutional neural network (CNN).

## Neural Network Architecture
- **CNN Architecture**: The neural network comprises several convolutional layers followed by batch normalization layers and fully connected layers. This setup is designed to extract and process spatial hierarchies in the input frames.

## Agent Implementation
- **Memory**: A deque used as memory for storing experiences, enabling experience replay.
- **Act Method**: Determines the action to take based on the current policy (epsilon-greedy).
- **Learning**: Updates the network weights based on sampled experiences from memory using the Adam optimizer.

## Training Loop
- **Epsilon Decay**: Epsilon value (exploration factor) decreases over time, favoring exploitation of learned values.
- **Training Episodes**: The agent is trained over multiple episodes, improving its policy based on the received rewards and updating the target network periodically.

## Visualization
- **Video Generation**: Post training, the code can generate a video showing the agent playing Ms. Pac-Man.