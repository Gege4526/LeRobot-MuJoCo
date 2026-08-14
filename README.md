# Training ACT, SmolVLA, and Pi0 with LeRobot in MuJoCo
## ACT — Action Chunking with Transformers

**ACT (Action Chunking with Transformers)** is an **imitation learning** method designed for robotic manipulation tasks.

### 1. Action Chunking

In ACT, the **chunk size** is initially set to a fixed value \(k\).

Every \(k\) timesteps, the agent receives the current observation and predicts the **next \(k\) actions** at once. These predicted actions are then executed sequentially.

### 2. Temporal Ensembling

For each timestep, the same action can be predicted up to **\(k\) times** from different action chunks.

These predictions are combined using a **weighted average**, with more recent predictions receiving higher weights.

<p align="center">
  <img src="Assets/algo.jpg" width="800">
</p>


<p align="center">
  <img src="Assets/detail_architecture.jpg" width="800">
</p>

**Paper**
https://arxiv.org/abs/2304.13705

