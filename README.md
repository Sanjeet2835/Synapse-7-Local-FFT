# Synapse-7: AI-Generated Image Detection using Frequency Domain Analysis

# 1. Project Abstract

**Synapse-7** is a deep learning classifier designed to distinguish between **AI-generated synthetic imagery** and **real images**. Unlike traditional CNNs that rely solely on spatial pixel patterns, this project implements a **Frequency Domain Analysis** pipeline to detect the spectral artifacts often left behind by generative models (GANs and Diffusion models).

By transforming input images into frequency maps using **Local Patch-wise Fast Fourier Transforms (FFT)**, the model exposes high-frequency irregularities invisible to the human eye. These spectral features are then processed by a custom-tuned **ResNet-34** backbone.

### **Key Highlights:**

* **Methodology:** Hybrid approach combining Signal Processing (FFT) with Deep Learning (CNNs).
* **Performance:** Achieved **97.3% Accuracy** and **97.3% F1-Score** on the validation set.
* **Architecture:** ResNet-34 backbone with frozen early layers to preserve low-level feature extraction, fine-tuned on spectral data.
* **Tech Stack:** PyTorch Lightning, MLFlow (Experiment Tracking), and Custom FFT Preprocessing.


# 2. System Architecture

The core innovation of Synapse-7 is its hybrid approach to feature extraction. Instead of relying solely on spatial artifacts (which modern generators are getting better at hiding), this system analyzes the **frequency spectrum** of the image to detect structural irregularities common in upsampling layers of GANs and Diffusion models.

### **A. The Frequency Domain Pipeline (`LocalPatchFFT`)**

Before entering the neural network, every image undergoes a custom signal processing transformation designed to highlight "spectral fingerprints."

1. **Patch Extraction:** The high-resolution input is divided into local `32x32` patches.
2. **2D Fast Fourier Transform (FFT):** Each patch is converted from the Spatial Domain to the Frequency Domain using `torch.fft.fft2`.
3. **Log-Magnitude Spectrum:** We compute the logarithmic magnitude of the complex frequencies to visualize high-frequency artifacts while compressing the dynamic range.
4. **Reassembly:** The processed patches are stitched back together to form a "Frequency Map" tensor, which serves as the new input for the CNN.

### **B. Model Backbone (ResNet-34)**

The classifier is built on a **ResNet-34** architecture pretrained on ImageNet, adapted specifically for spectral data.

* **Transfer Learning Strategy:**
* **Frozen Layers:** The initial stem (`conv1`) and the first residual block (`layer1`) are frozen. This preserves the network's ability to detect fundamental low-level features (edges, textures) without overfitting to the new domain immediately.
* **Trainable Layers:** Deeper layers (`layer2` through `layer4`) are fine-tuned to learn the specific high-level frequency patterns of AI-generated content.


* **Classification Head:** The default ImageNet head is replaced with a custom binary classification block:
* `Linear (512 -> 128)` + `ReLU`
* `Dropout (p=0.5)` (for regularization)
* `Linear (128 -> 1)` + `Sigmoid`


# 3. Dataset & Pipeline

The system utilizes a custom `MultiGenDataset` class designed to aggregate training data from multiple diverse sources (e.g., Midjourney, DALL-E, Stable Diffusion) into a single unified stream. This ensures the model learns generalized artifacts rather than overfitting to a specific generator's style.

### **A. Data Directory Structure**

To reproduce the training results, the dataset must be organized hierarchically. The data loader iterates through every sub-directory in the root `Data/` folder, treating each as an independent source.

```text
Data/
├── [Generator_Source_A] (e.g., Midjourney)
│   ├── train
│   │   ├── ai        # Synthetic Images (Label: 1)
│   │   └── nature    # Real Images (Label: 0)
│   └── val
│       ├── ai
│       └── nature
├── [Generator_Source_B] (e.g., Stable_Diffusion)
│   ├── train...
│   └── val...

```

### **B. Preprocessing Pipeline**

Before Frequency Domain analysis, raw images undergo a standardized transformation pipeline using `torchvision.transforms`:

1. **Resize & Crop:** Images are resized to `256px` and center-cropped to `224x224` to match the ResNet input requirements while maintaining aspect ratio consistency.
2. **Normalization:** Pixel values are scaled to the standard ImageNet mean and standard deviation:
* `mean=[0.485, 0.456, 0.406]`
* `std=[0.229, 0.224, 0.225]`


3. **Tensor Conversion:** Converted to PyTorch tensors (`C x H x W`).

### **C. Data Module**

The training logic is encapsulated in a `LightningDataModule`, which handles:

* **Batching:** Default batch size of `96` 
* **Workers:** Utilizes `multiprocessing` (4 workers) for efficient data loading to prevent GPU bottlenecks.


# 4. Training Configuration

The training workflow is managed by **PyTorch Lightning**, which handles the training loop, validation integration, and hardware acceleration. The configuration focuses on preventing overfitting—a common challenge when training on frequency domain features.

### **A. Hyperparameters**

| Parameter | Value | Description |
| --- | --- | --- |
| **Batch Size** | `96` | Tuned for maximum throughput on a 6GB VRAM RTX 3050. |
| **Learning Rate** | `1e-3` | Initial learning rate for the AdamW optimizer. |
| **Max Epochs** | `50` | The theoretical upper limit (training usually stops earlier). |
| **Precision** | `16-mixed` | Uses FP16 mixed precision for faster training and lower memory usage. |
| **Weight Decay** | `1e-2` | Regularization applied to prevent the model from exploding weights. |

### **B. Optimization Strategy**

* **Optimizer:** **AdamW** (Adam with Decoupled Weight Decay). This was chosen over standard SGD because it converges faster on complex spectral features.
* **Loss Function:** **Binary Cross-Entropy with Logits** (`BCEWithLogitsLoss`). This combines a Sigmoid layer and the BCELoss in one single class, which is numerically more stable than using a plain Sigmoid followed by BCELoss.

### **C. Callbacks & Safety Nets**

To ensure the best possible model is saved:

1. **Early Stopping:**
* **Monitor:** `val_f1` (Maximizing F1-Score).
* **Patience:** `4` epochs.
* *Effect:* If the model's F1 score stops improving for 4 epochs, training stops immediately to prevent overfitting.


2. **Learning Rate Scheduler (`ReduceLROnPlateau`):**
* **Monitor:** `val_loss`.
* **Factor:** `0.1` (Reduces LR by 10x).
* **Patience:** `2` epochs.
* *Effect:* If the loss plateaus, the learning rate drops, allowing the model to make finer adjustments to the weights.


# 5. Evaluation & Metrics

The model was evaluated on a held-out validation set of **7,000 images** (20% split). Training automatically halted at **Epoch 16** via Early Stopping when the F1-Score saturated, ensuring peak generalization.

### **A. Performance Summary**

| Metric | Score | Interpretation |
| --- | --- | --- |
| **Accuracy** | **97.3%** | Correctly classified nearly all samples. |
| **F1-Score** | **97.3%** | Harmonic mean of Precision and Recall; shows balanced performance. |
| **Precision** | **98.1%** | **Low False Positive Rate.** When the model says "AI", it is highly likely to be true. It rarely flags real art as fake. |
| **Recall** | **96.5%** | **High Detection Rate.** The model catches the vast majority of AI-generated content, missing very few. |

### **B. Diagnostic Plots**

To verify reliability beyond simple accuracy, two key visualizations are generated during inference:

1. **Confusion Matrix:**
* Visualizes the exact count of *False Positives* (Real flagged as AI) vs. *False Negatives* (AI flagged as Real).
* *Result:* The model shows a slight bias towards Precision, which is preferred in detection systems to avoid accusing human artists falsely.


2. **Calibration Curve (Reliability Diagram):**
* Plots *Predicted Confidence* vs. *True Probability*.
* *Result:* The curve follows the diagonal , indicating the model's confidence score (e.g., "80% sure this is AI") actually corresponds to an 80% real-world probability.


### **C. Model Artifacts**

* **Best Checkpoint:** Saved as `resnet34_localfft_v1.pth` (PyTorch state dictionary).
* **Tracking:** All metrics, hyperparameters, and artifacts are logged to **MLFlow** for experiment versioning.



# 6. Limitations & Future Work

While Synapse-7 achieves high accuracy on the validation set, the current implementation operates under specific constraints. This section outlines the boundaries of the system and identifies key areas for scaling.

### **A. Resource Constraints & Scope**

1. **Dataset Reduction:**
* The model was trained on the **"Tiny GenImage"** subset (~35,000 images) rather than the full **GenImage Benchmark** (which contains >1 million pairs). While sufficient for proof-of-concept, this reduced scope limits the model's exposure to the long-tail diversity of the full dataset.


2. **Backbone Efficiency vs. Capacity:**
* We deliberately selected **ResNet-34** over larger variants (like ResNet-50 or ResNet-101).
* *Rationale:* Given the hardware constraints (single GPU), ResNet-34 offered the optimal trade-off between training speed and feature extraction capability, preventing bottlenecks without sacrificing convergence stability.


3. **Hyperparameter Heuristics:**
* Due to hardware limitations, extensive **Neural Architecture Search (NAS)** and automated hyperparameter tuning (e.g., Optuna sweeps) were not performed. The current configuration relies on heuristic best practices (e.g., standard AdamW defaults) rather than empirically optimized values.



### **B. Future Scope**

1. **Scale-Up:**
* Train on the full **GenImage** dataset using multi-GPU distributed training (DDP) to validate performance at scale.


2. **Advanced Architectures:**
* Experiment with **Swin Transformers** or **ConvNeXt**, which may capture global frequency dependencies better than standard CNNs.









