# 🖼️ Image Captioning with Attention

A deep learning model that automatically generates natural language captions for images using a **CNN encoder + LSTM decoder with Bahdanau Attention**. Trained on the Flickr8k dataset using PyTorch.

---

## 📌 Overview

This project implements an image captioning pipeline from scratch, combining:
- **ResNet-50** as a frozen CNN encoder to extract spatial image features
- **Bahdanau (Additive) Attention** to focus on relevant image regions at each decoding step
- **LSTM Decoder** that generates captions word-by-word, guided by attention

At inference time, the model takes an image and outputs a descriptive caption like *"a dog running through a grassy field"*.

---

## 🏗️ Architecture

```
Image
  │
  ▼
ResNet-50 (frozen, removes avg pool + FC)
  │  → [batch, 49, 2048]  (7×7 spatial feature map)
  ▼
Attention Module  ←──────────────────────────┐
  │  Computes context vector                 │
  │  α = softmax(W · tanh(Wh·h + Wf·f))     │
  ▼                                          │
LSTM Cell  ──── hidden state h_t ───────────┘
  │
  ▼
Linear → Softmax → Word prediction
```

---

## 📂 Project Structure

```
image2text/
├── image2text.ipynb      # Main training & inference notebook
├── requirements.txt      # Python dependencies
├── .gitignore            # Files excluded from version control
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/image2text.git
cd image2text
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

This project uses the **Flickr8k** dataset. You can download it from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).

Place the files so your directory looks like:

```
data/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   └── ...
└── captions.txt
```

### 4. Run the notebook

Open `image2text.ipynb` in Jupyter or on [Kaggle](https://www.kaggle.com/) (recommended for free GPU access) and run all cells.

---

## ⚙️ Model Configuration

| Hyperparameter     | Value  |
|--------------------|--------|
| Encoder            | ResNet-50 (pretrained, frozen) |
| Encoder output dim | 2048   |
| Attention dim      | 256    |
| Decoder (LSTM) dim | 512    |
| Embedding size     | 256    |
| Vocabulary freq threshold | 5 |
| Dropout            | 0.5    |
| Optimizer          | Adam   |
| Epochs             | 10     |
| Batch size         | 32     |
| Device             | CUDA (T4 GPU) |

---

## 🧠 Key Components

### `Vocabulary`
Builds a word-to-index mapping from captions. Words appearing fewer than `freq_threshold` times are mapped to `<UNK>`. Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`.

### `ImageCaptionDataset`
Custom PyTorch `Dataset` that loads images and numericalizes their captions. Uses `MyCollate` to pad caption sequences within each batch.

### `EncoderCNN_Attention`
Wraps ResNet-50, strips the final pooling and FC layers, and returns a **7×7 spatial feature map** (reshaped to 49 vectors of size 2048).

### `AttentionModule`
Implements **Bahdanau (additive) attention**. At each decoder timestep, scores are computed over all 49 image regions using the current hidden state, producing a context vector.

### `DecoderRNN_Attention`
An LSTM-based decoder that:
- Initializes hidden/cell state from the mean image features
- At each step, attends over image features and concatenates the context with the word embedding before the LSTM cell

---

## 📊 Training

The training loop runs for 10 epochs using **CrossEntropyLoss** (ignoring `<PAD>` tokens) and the **Adam optimizer**. Loss is printed per epoch to monitor convergence.

```
Epoch [1/10], Loss: 3.4521
Epoch [2/10], Loss: 2.9812
...
```

---

## 🔍 Inference

Given an image, the model generates a caption greedily (token-by-token) until `<EOS>` is produced or a max length is reached. The generated caption is displayed alongside the input image.

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full list. Core dependencies:

- `torch` / `torchvision`
- `Pillow`
- `pandas`
- `matplotlib`

---

## 🙏 Acknowledgements

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) — Kaggle
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) — Xu et al. (2015), the paper this attention mechanism is based on
- [PyTorch](https://pytorch.org/) — deep learning framework

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
