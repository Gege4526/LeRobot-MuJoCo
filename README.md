# Training ACT with LeRobot in MuJoCo

## Table of Contents
- [Setup](#setup)
- [1. Collect Demonstration Data](#1-collect-demonstration-data)
- [2. Playback Your Data](#2-playback-your-data)
- [3. Train Action Chunking Transformer (ACT)](#3-train-action-chunking-transformer-act)
- [📚 Theory](#-theory)
  - [➡️ ACT — Action Chunking with Transformers](#️-act--action-chunking-with-transformers)
    - [1. Action Chunking](#1-action-chunking)
    - [2. CVAE](#2-cvae)
    - [3. Temporal Ensembling](#3-temporal-ensembling)
    - [Reference](#reference)
  - [➡️ Pi0](#️-pi0)
    - [1. Pre-trained VLM](#1-pre-trained-vlm)
    - [2. Action Expert](#2-action-expert)
    - [Reference](#reference-1)
  - [➡️ SmolVLA](#️-smolvla)
    - [Reference](#reference-2)

## Setup
```bash
conda create -n py310 python=3.10
conda activate py310
```
```bash
git clone git@github.com:Gege4526/LeRobot-MuJoCo.git
cd ~/LeRobot-MuJoCo
pip install -r requirements.txt
conda install jupyterlab
pip install ipywidgets ipykernel
python -m ipykernel install --user --name py310 --display-name "py310"
code .
```
```bash
cd asset/objaverse
unzip plate_11.zip
```
## 1. Collect Demonstration Data

Run the `1.collect_data.ipynb` notebook to collect demonstration data in the provided environment.

The task is to **pick up the cup and place it onto the plate**. The environment considers the task successful when:

* The cup is placed on the plate.
* The gripper is open.
* The end-effector is positioned above the cup.

### 🎮 Keyboard Controls

* `W/A/S/D`: Move along the **x-y plane**
* `R/F`: Move along the **z-axis**
* `Q/E`: Tilt the end-effector
* `Arrow Keys`: Rotate the end-effector
* `Space`: Toggle the gripper state
* `Z`: Reset the environment and discard the cached data from the current episode

### 🖥️ Observation Views

The rendered observation contains four views:

* **Top-right:** Agent view
* **Bottom-right:** First-person wrist camera view
* **Top-left:** Side view
* **Bottom-left:** Top-down view

### Data Structure
```bash
fps = 20,
features={
    "observation.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["state"], # x, y, z, roll, pitch, yaw
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"], # 6 个关节角 + 1 个夹爪
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["obj_init"], # 仅物体初始位置，训练中不使用
    },
},
```

## 2. Playback Your Data

Run `2.visualize_data.ipynb` to replay and inspect the collected demonstration data.

The main MuJoCo window reconstructs the simulation scene and replays the recorded robot actions.

The overlaid images show the observations stored in the dataset:

* **Top-right:** Agent-view image from the dataset
* **Bottom-right:** Wrist-camera image from the dataset

This allows you to verify that the recorded actions and visual observations are correctly synchronized.

## 3. Train Action Chunking Transformer (ACT)

Run `3.train.ipynb` to train an **Action Chunking Transformer (ACT)** on the collected demonstration dataset.

Training takes approximately **30–60 minutes**, depending on your hardware and dataset size.

In this example, ACT is trained with:

```python
chunk_size = 10
```

The trained checkpoint will be saved to:

```text
./ckpt/act_y
```

## 📚 Theory
### ➡️ ACT — Action Chunking with Transformers

**ACT (Action Chunking with Transformers)** is an **imitation learning** method designed for robotic manipulation tasks.

#### 1. Action Chunking

ACT predicts a **chunk of (k) future actions** instead of predicting only one action at a time.

$$
A_t=[a_t,a_{t+1},...,a_{t+k-1}]
$$

#### 2. CVAE

ACT uses a **Conditional Variational Autoencoder (CVAE)** to model variations in expert demonstrations.

During training, the expert action sequence is encoded into a **latent variable (z)**, which captures different possible action styles or behaviors.

$$
\text{Demonstration} \rightarrow \text{Encoder} \rightarrow z
$$

The Transformer then predicts the action chunk conditioned on the observation and (z):

$$
(\text{Observation},z) \rightarrow \text{Action Chunk}
$$

This helps reduce **behavior averaging** when multiple valid actions exist for the same observation.

> [!IMPORTANT]
> **CVAE** → models variations in expert demonstrations and helps handle multimodal actions.

#### 3. Temporal Ensembling

For each timestep, the same action can be predicted multiple times from overlapping action chunks.

These predictions are combined using a **weighted average**, with more recent predictions receiving higher weights.

<p align="center">
  <img src="Assets/algo.jpg" width="900">
</p>

> [!NOTE]
> **Training:**  
> Expert Actions → CVAE Encoder → $z$  
> Image Features (CNN) + Joint State + $z$ → Transformer → Action Chunk
>
> **Inference:**  
> Image Features (CNN) + Joint State + $z=0$ → Transformer → Action Chunk

<p align="center">
  <img src="Assets/detail_architecture.jpg" width="900">
</p>

#### Reference
**Paper**
https://arxiv.org/abs/2304.13705

### ➡️ Pi0 

**π₀** is a **VLA** model developed by Physical Intelligence for general-purpose robot control.

Unlike ACT, which directly predicts action chunks, π₀ combines a pretrained **VLM** with a specialized **action expert** to generate continuous robot actions.

<p align="center">
  <img src="Assets/overview_pi0.jpg" width="900">
</p>

#### 1. Pre-trained VLM
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

> [!IMPORTANT]
> π₀ extends **PaliGemma** with three main modifications:
>
> 1. **Robot State & Action Projection** – maps continuous robot states and actions into Transformer representations.
> 2. **Flow-Matching Time Embedding** – encodes the flow timestep $\tau$ using an additional MLP.
> 3. **Action Expert** – a smaller Transformer specialized for continuous robot action generation.


#### 2. Action Expert
conditional flow matching
action chunk

#### Reference
**Paper**
https://arxiv.org/abs/2410.24164

**Blog**
https://www.pi.website/blog/pi0

### ➡️ SmolVLA

#### Reference
**Paper**https://arxiv.org/abs/2506.01844

**Blog**
https://huggingface.co/blog/smolvla



