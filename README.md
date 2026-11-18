# Vision-Language Models with CLIP

**Zero-shot learning, image-text retrieval, parameter-efficient fine-tuning, and LoRA adaptation with PyTorch**

Developed as part of the IT5005 Artificial Intelligence course project at the National University of Singapore.

## Overview

CLIP (Contrastive Language-Image Pre-training) learns a shared representation space for images and text, allowing visual inputs and natural-language descriptions to be compared directly.

This project investigated CLIP from two perspectives:

1. **Part 1 — Frozen CLIP:** How well can a pretrained CLIP model perform image classification without task-specific training, and how much can performance improve by training a classifier on frozen CLIP features?
2. **Part 2 — Adapted CLIP:** How effectively can CLIP be adapted for image-text retrieval using parameter-efficient fine-tuning strategies while keeping most pretrained parameters frozen?

The project progressed from zero-shot inference to supervised probing, contrastive image-text training, projection-only fine-tuning, and LoRA-based adaptation.

## Key Results

| Task | Method | Test Performance |
|---|---|---:|
| ImageNet-1k classification | Zero-shot CLIP | **62.19% accuracy** |
| ImageNet-1k classification | Frozen CLIP + trained probe | **69.33% accuracy** |
| Flickr8k retrieval | Zero-shot CLIP | **0.7424 mean recall** |
| Flickr8k retrieval | Projection-only + Triplet | **0.7874 mean recall** |
| Flickr8k retrieval | Projection-only + InfoNCE | **0.7994 mean recall** |
| Flickr8k retrieval | LoRA r=2 + projections | **0.8107 mean recall** |

The trained classifier improved ImageNet test accuracy from **62.19% to 69.33%**, while the selected LoRA configuration improved Flickr8k test mean recall from **0.7424 to 0.8107**.

---

# Part 1 — Frozen CLIP

## Dataset

Part 1 used ImageNet-1k with:

- **35,000 training images**
- **5,000 validation images**
- **10,000 test images**
- **1,000 image classes**

The pretrained model was:

`openai/clip-vit-base-patch32`

CLIP maps both image and text inputs into a shared **512-dimensional embedding space**.

## Zero-Shot Classification

For zero-shot classification, ImageNet class names were converted into natural-language prompts and encoded using CLIP's frozen text encoder.

Each image was encoded using the frozen vision encoder, and predictions were obtained by comparing image and text embeddings.

No ImageNet labels were used to update CLIP parameters during zero-shot inference.

### Prompt Engineering

Ten prompt templates were evaluated using the validation set.

The best observed validation result was:

> **"a blurry, low-quality photo of a {}." — 60.74% validation accuracy**

Other leading prompts were very close:

| Prompt | Validation Accuracy |
|---|---:|
| `a blurry, low-quality photo of a {}.` | **60.74%** |
| `a photo of a {}.` | 60.68% |
| `an image of a {}` | 60.56% |

The differences between the leading prompts were small. For example, the difference between the first and third prompts corresponded to only nine net additional correct predictions among 5,000 validation images.

An unrelated story-style prompt reduced validation accuracy to **53.00%**, illustrating that prompt wording can substantially affect CLIP's zero-shot predictions.

## Cosine Similarity vs Raw Dot Product

The project also compared two methods for matching image and text embeddings.

On the same 10,000 ImageNet test images:

| Similarity Measure | Test Accuracy |
|---|---:|
| Raw dot product | **38.21%** |
| Cosine similarity | **62.19%** |

The two methods produced different predicted classes for **51.90%** of the test images.

For image vector `i` and text vector `t`:

```text
i · t = ||i|| × ||t|| × cos(θ)
```

For a fixed image, the image norm multiplies all candidate class scores by the same positive constant and therefore cannot change their ranking.

Text norms, however, vary between candidate classes and can change the ranking under the raw dot product.

L2 normalization removes this effect and converts the dot product into cosine similarity.

## Frozen-Feature Probe

A supervised classifier was then trained on frozen CLIP image features.

CLIP itself remained frozen; only the classifier head was trained.

The dataset was separated into:

```text
Training set     → learn classifier parameters
Validation set   → select model checkpoint
Test set         → final evaluation only
```

The highest validation accuracy occurred at:

> **Epoch 7 — 68.42%**

The selected epoch-7 model achieved:

> **69.33% test accuracy**

compared with:

> **62.19% zero-shot accuracy**

This represents a **7.14 percentage-point improvement**.

The trade-off is reduced flexibility: unlike zero-shot CLIP, the trained classifier is tied to the 1,000 classes on which it was trained.

---

# Part 2 — CLIP Adaptation for Image-Text Retrieval

## Dataset

Part 2 used Flickr8k image-caption pairs:

- **6,000 training pairs**
- **1,000 validation pairs**
- **1,091 test pairs**

The task changed from classification to **bidirectional image-text retrieval**.

Two retrieval directions were evaluated:

- **Image-to-Text (I2T):** retrieve the paired caption for an image
- **Text-to-Image (T2I):** retrieve the paired image for a caption

## Evaluation Metrics

Retrieval performance was evaluated using:

- Recall@1
- Recall@5
- Recall@10

for both image-to-text and text-to-image retrieval.

The primary summary metric was **Mean Recall**, calculated across R@1, R@5, and R@10 in both retrieval directions.

---

## Zero-Shot Retrieval Baseline

Before fine-tuning, pretrained CLIP achieved:

| Metric | Score |
|---|---:|
| I2T R@1 | 0.5298 |
| I2T R@5 | 0.8130 |
| I2T R@10 | 0.8937 |
| T2I R@1 | 0.5380 |
| T2I R@5 | 0.7919 |
| T2I R@10 | 0.8882 |
| **Mean Recall** | **0.7424** |

This provided the baseline for evaluating subsequent adaptation strategies.

---

# Contrastive Training Objectives

## InfoNCE

InfoNCE compares every image with every caption within a batch.

For a batch of 64 aligned image-caption pairs, each example has:

> **1 positive + 63 in-batch negatives per retrieval direction**

Image and text features are normalized before computing their similarity matrix.

The implemented bidirectional loss can be represented as:

```text
L = 1/2 [
    CE(S / τ, y)
    +
    CE(Sᵀ / τ, y)
]
```

where:

- `S` is the image-text similarity matrix
- `τ` is the temperature
- `y` identifies the matching positions

The project used a temperature of:

> **τ = 0.01**

The two directions correspond to:

```text
Image → Caption
Caption → Image
```

and their cross-entropy losses are averaged.

## Triplet Loss

The Triplet implementation instead identifies the hardest mismatched example for each query.

With a batch size of 64:

- InfoNCE uses all **63 in-batch negatives**
- Triplet explicitly uses **1 hardest negative**

The Triplet objective used a margin of:

> **0.2**

This allowed the project to compare two different learning signals for image-text alignment.

---

# Projection-Only Parameter-Efficient Fine-Tuning

The first adaptation strategy froze both CLIP encoders and trained only:

- visual projection
- text projection

This resulted in:

> **655,360 trainable parameters**

or approximately:

> **0.43% of CLIP**

The encoders therefore continued producing fixed pretrained features, while the projection layers learned a task-specific mapping into the shared embedding space.

## InfoNCE vs Triplet Results

Using projection-only fine-tuning:

| Method | I2T R@1 | T2I R@1 | Mean Recall |
|---|---:|---:|---:|
| Triplet | 0.5857 | 0.5885 | **0.7874** |
| InfoNCE | 0.6169 | 0.5811 | **0.7994** |

InfoNCE achieved the higher overall mean recall in this experiment, although the comparison was mixed by retrieval direction.

---

# Learning-Rate and Checkpoint Selection

Model and hyperparameter choices were made using the validation set rather than the final test set.

Three learning rates were compared:

- `1e-6`
- `1e-5`
- `1e-4`

The selected learning rate was:

> **1e-5**

with a validation mean recall of:

> **0.8168**

The `1e-6` configuration adapted more slowly, while the `1e-4` configuration peaked early and subsequently declined.

This illustrated the importance of selecting hyperparameters based on held-out validation performance rather than training loss alone.

## Early Stopping and Overfitting

A longer diagnostic run demonstrated that continued optimization of training loss did not necessarily improve validation retrieval performance.

Training loss decreased from:

> **0.3418 at epoch 1**

to:

> **0.0695 at epoch 25**

while validation mean recall peaked earlier and later declined.

This provided a practical example of why checkpoint selection and early stopping are important when adapting pretrained models.

---

# LoRA Adaptation

The second parameter-efficient strategy introduced **Low-Rank Adaptation (LoRA)** into selected attention layers of CLIP's text encoder while continuing to train the projection layers.

The vision encoder remained frozen.

For a frozen weight matrix `W₀`, the LoRA update can be represented as:

```text
y = W₀x + (α / r)BAx
```

where:

- `W₀` remains frozen
- `A` and `B` are trainable low-rank matrices
- `r` is the LoRA rank
- `α / r` controls the update scale

LoRA was applied to the:

- query projections (`q_proj`)
- value projections (`v_proj`)

across the 12 text-transformer layers.

This resulted in **24 adapted projection layers**.

## LoRA Initialization and Merge

The implementation initialized:

- `A` randomly using Kaiming-uniform initialization
- `B` to zero

Because `B = 0` initially:

```text
BA = 0
```

and the adapted model initially behaves identically to the pretrained model.

After training, the low-rank update can be merged into the frozen base weight:

```text
W₀ ← W₀ + (α / r)BA
```

The project verified that retrieval performance remained unchanged before and after merging.

For the selected model:

> Validation mean recall before merge: **0.8253**  
> Validation mean recall after merge: **0.8253**

---

# LoRA Rank Selection

Three LoRA ranks were evaluated:

| Rank | Best Validation Mean Recall | LoRA Parameters |
|---|---:|---:|
| **r = 2** | **0.8253** | 49,152 |
| r = 8 | 0.8245 | 196,608 |
| r = 32 | 0.8237 | 786,432 |

Higher ranks reduced training loss but did not improve validation mean recall.

The selected configuration was therefore:

> **LoRA r = 2**

With the projection layers included, this model trained:

> **704,512 parameters**

or approximately:

> **0.47% of CLIP's 151 million parameters**

This illustrates an important result from the project: **more trainable capacity did not necessarily produce better held-out retrieval performance**.

---

# Final Test Comparison

Each selected model was evaluated once on the held-out set of 1,091 image-caption pairs.

| Method | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Mean Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 0.5298 | 0.8130 | 0.8937 | 0.5380 | 0.7919 | 0.8882 | **0.7424** |
| Projection-only + Triplet | 0.5857 | 0.8488 | 0.9203 | 0.5885 | 0.8579 | 0.9230 | **0.7874** |
| Projection-only + InfoNCE | 0.6169 | 0.8662 | 0.9340 | 0.5811 | 0.8634 | 0.9349 | **0.7994** |
| **LoRA r=2 + projections** | **0.6288** | **0.8818** | **0.9450** | **0.5958** | **0.8735** | **0.9395** | **0.8107** |

The selected LoRA model achieved the highest observed mean recall:

> **0.8107**

compared with the zero-shot baseline:

> **0.7424**

It also improved image-to-text R@1 by approximately:

> **+9.9 percentage points**

over zero-shot CLIP.

However, the smaller differences between fine-tuning strategies should be interpreted cautiously because each configuration was evaluated using a single experimental run.

---

# Qualitative Retrieval Analysis

Quantitative retrieval metrics do not fully capture the behavior of a vision-language model.

Qualitative inspection showed that CLIP could identify semantically similar scenes while still failing exact-pair retrieval.

For example, multiple captions describing visually similar brown dogs running outdoors received very similar similarity scores.

This highlights an important distinction:

> **Semantic similarity is not always equivalent to exact image-caption pairing.**

Retrieval could also succeed in one direction while failing in the reverse direction.

These examples helped identify limitations that were not obvious from aggregate Recall@K metrics alone.

---

# Limitations

Several limitations should be considered when interpreting the results.

### Single Runs

Most configurations were evaluated using one run per setting.

Small observed differences therefore cannot establish that one configuration would consistently outperform another across random seeds.

Future evaluation should include:

- repeated experiments across multiple random seeds
- paired bootstrap confidence intervals

### Frozen Encoders

Projection-only fine-tuning cannot modify the features extracted by either encoder.

The LoRA configuration adapts selected text-attention layers, but the vision encoder remains frozen.

Therefore, information not represented effectively by the frozen vision encoder cannot be recovered by these adaptation strategies.

### Generalization

Fine-tuning on Flickr8k may affect CLIP's original zero-shot capabilities or transfer performance on other datasets.

This project did not directly evaluate that trade-off.

Future work could include:

- comparing ImageNet zero-shot performance before and after adaptation
- evaluating retrieval on Flickr30k or COCO
- testing robustness across different caption styles and datasets

---

# My Contributions

This was a team-based AI project.

My work included completing and analyzing the **Part 2 image-text retrieval and CLIP adaptation workflow**, including:

- Worked with the Flickr8k image-text retrieval pipeline using pretrained CLIP.
- Implemented and evaluated bidirectional retrieval using **Recall@1, Recall@5, and Recall@10**.
- Worked with **projection-only parameter-efficient fine-tuning** of CLIP.
- Implemented and analyzed **InfoNCE and Triplet loss** for contrastive image-text learning.
- Implemented and evaluated **LoRA-based adaptation** of selected CLIP text-attention layers.
- Conducted learning-rate comparison, checkpoint selection, early-stopping analysis, and LoRA rank selection.
- Compared zero-shot, projection-only, and LoRA-adapted retrieval performance.
- Conducted quantitative and qualitative analysis of retrieval results and model limitations.
- Presented the project's **methodology, final results, uncertainty in model comparisons, and limitations**.

---

# Tech Stack

### Deep Learning

- Python
- PyTorch
- torchvision
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA

### Data & Evaluation

- NumPy
- scikit-learn
- Matplotlib
- Pillow
- tqdm
- TensorBoard

### Development

- Jupyter Notebook

---

# Reproducibility

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

Core requirements include:

```text
torch>=2.4
torchvision>=0.19
transformers>=5.0,<6
datasets>=3.0
numpy
scikit-learn>=1.4
matplotlib
pillow
requests
tqdm
tensorboard
jupyter
ipykernel
ipywidgets
peft
```

The project uses:

> `openai/clip-vit-base-patch32`

as the pretrained CLIP checkpoint.

---

# Project Structure

```text
clip-vision-language-modeling/
│
├── Part_1.ipynb
├── Part_2.ipynb
├── requirements.txt
└── README.md
```

**Part 1** covers frozen CLIP, zero-shot ImageNet classification, prompt comparison, similarity measures, and supervised probing.

**Part 2** covers Flickr8k image-text retrieval, contrastive losses, projection-only fine-tuning, LoRA adaptation, hyperparameter selection, and retrieval evaluation.

---

# Key Takeaways

This project demonstrated several practical properties of adapting pretrained vision-language models:

1. **Zero-shot CLIP provides strong performance without task-specific training.**

2. **Task-specific supervision can improve performance**, but may reduce the flexibility that makes zero-shot models attractive.

3. **Similarity design matters.** Cosine similarity substantially outperformed the raw dot product for ImageNet zero-shot classification in this experiment.

4. **InfoNCE and Triplet loss provide different training signals.** InfoNCE used all in-batch negatives and achieved higher overall retrieval performance in this run.

5. **Parameter-efficient adaptation can substantially improve retrieval.** LoRA plus projection fine-tuning increased Flickr8k mean recall from **0.7424 to 0.8107** while training less than 1% of CLIP's parameters.

6. **More parameters are not necessarily better.** Increasing LoRA rank reduced training loss but did not improve validation mean recall.

7. **Validation-based model selection is essential.** Learning-rate selection, checkpoint selection, early stopping, and LoRA rank selection were all based on held-out validation performance rather than the test set.

8. **Small performance differences require caution.** Repeated seeds and paired statistical evaluation would be needed to determine whether the smaller observed gaps between adaptation strategies are stable.

---

# Project Context

- **Course:** IT5005 Artificial Intelligence
- **Institution:** National University of Singapore
- **Project Type:** Team-based course project
- **Tasks:** Image classification and image-text retrieval
- **Datasets:** ImageNet-1k and Flickr8k
- **Framework:** PyTorch / Hugging Face
- **Status:** Completed, November 2026
