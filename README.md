# Training ACT, SmolVLA, and Pi0 with LeRobot in MuJoCo
## 🤖 ACT — Action Chunking with Transformers

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

### Reference
**Paper**
https://arxiv.org/abs/2304.13705

## 🤖 π₀ 

**π₀** is a **VLA** model developed by Physical Intelligence for general-purpose robot control.

Unlike ACT, which directly predicts action chunks, π₀ combines a pretrained **VLM** with a specialized **action expert** to generate continuous robot actions.

<p align="center">
  <img src="Assets/overview_pi0.jpg" width="800">
</p>

### 1. Pre-trained VLM
π₀ builds its VLM backbone on **PaliGemma**, which consists of:

* **SigLIP**: A **ViT (Vision Transformer)** used as the vision encoder. It converts input images into image tokens. A **linear projection layer** then maps these image tokens to the same embedding dimension as the Gemma text tokens, allowing them to be processed together.

* **Gemma**: A **decoder-only Transformer** used as the language-model backbone. It jointly processes the projected image tokens and text tokens to produce contextual visual-language representations.

> [!NOTE]
> **SigLIP**
> Image → Image Tokens
> 
> **Gemma**
> Image Tokens + Text Tokens → Contextual Representations

<p align="center">
  <img src="Assets/PaliGemma.jpg" width="800">
</p>

### 2. Action Expert


### Reference
**Paper**
https://arxiv.org/abs/2410.24164

**Blog**
https://www.pi.website/blog/pi0

## 🤖 SmolVLA

### Reference
**Paper**https://arxiv.org/abs/2506.01844

**Blog**
https://huggingface.co/blog/smolvla



