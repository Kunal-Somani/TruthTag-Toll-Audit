# 🧮 The Math Behind the Project: HOG + SVM Vehicle Classifier

> A deep dive into the mathematics powering Histogram of Oriented Gradients (HOG) feature extraction and Support Vector Machine (SVM) classification.

---

## Table of Contents

- [Part 1: The Eye — HOG Feature Extraction](#part-1-the-eye--hog-feature-extraction)
  - [Step 1: Calculating Gradients](#step-1-calculating-gradients)
  - [Step 2: Magnitude & Orientation](#step-2-magnitude--orientation)
  - [Step 3: Spatial Binning](#step-3-spatial-binning)
  - [Step 4: Block Normalization](#step-4-block-normalization)
- [Part 2: The Brain — SVM Classification](#part-2-the-brain--svm-classification)
  - [Step 1: The Hyperplane](#step-1-the-hyperplane)
  - [Step 2: Maximizing the Margin](#step-2-maximizing-the-margin)
  - [Step 3: The Kernel Trick](#step-3-the-kernel-trick)
- [Elevator Pitch Summary](#elevator-pitch-summary)

---

## 📖 Read the Full Story on Medium

Before diving into the math, here's the **"why"** and **"how"** behind this project — written by the team.

| | Blog | Author | What it covers |
|--|------|--------|----------------|
| 🛣️ | **[How We Built a "Truth-Based" Audit Layer for India's Highway Network](https://medium.com/@ujjwal264a/how-we-built-a-truth-based-audit-layer-for-indias-highway-network-3e63dd414eed)** | Ujjwal | The problem — FASTag's "Semantic Gap," revenue leakage, and the vision for a cross-modal verification system. *3 min read* |
| 🔬 | **[Visual Auditing for Intelligent Transport Systems: A Classical CV Approach to Fraud Detection](https://medium.com/@kunal120222/visual-auditing-for-intelligent-transport-systems-a-classical-computer-vision-approach-to-fraud-cc70c7620357)** | Kunal | The engineering — why we chose Classical CV over Deep Learning, the full HOG+SVM pipeline, and how we tackled real-world challenges like occlusion and the "Van vs. Pickup" dilemma. *4 min read* |

> **Recommended reading order:** Ujjwal's blog first (the *why*), then Kunal's (the *how*).

---

## Part 1: The Eye — HOG Feature Extraction

HOG does **not** look at raw pixel values. It looks at **rates of change (derivatives)** — the edges and textures that define shape.

---

### Step 1: Calculating Gradients

For every pixel `(x, y)`, we compute how rapidly intensity changes in the horizontal and vertical directions using the kernel `[-1, 0, 1]`:

**Horizontal Gradient:**

$$G_x(x,y) = I(x+1,\, y) - I(x-1,\, y)$$

**Vertical Gradient:**

$$G_y(x,y) = I(x,\, y+1) - I(x,\, y-1)$$

> These are first-order finite difference approximations of the image derivative.

---

### Step 2: Magnitude & Orientation

We convert the `(Gx, Gy)` pair into polar form — a **magnitude** (edge strength) and an **angle** (edge direction):

**Edge Magnitude:**

$$m(x,y) = \sqrt{G_x^2 + G_y^2}$$

**Edge Orientation:**

$$\theta(x,y) = \arctan\!\left(\frac{G_y}{G_x}\right)$$

| Value | Meaning |
|-------|---------|
| High `m` | Sharp edge (e.g. windshield frame) |
| Low `m`  | Flat surface (e.g. door panel) |

---

### Step 3: Spatial Binning (The Histogram)

Rather than tracking every pixel individually (noisy), we group pixels into **8×8 cells** and build a histogram.

- **9 orientation bins** covering angles: `0°, 20°, 40°, ..., 160°`
- Each pixel **votes** for its angle bin
- The **vote weight = magnitude** `m`

**Example:**
```
Strong vertical edge  → θ = 90°, m = 100  →  adds +100 to the 90° bin
Weak diagonal edge    → θ = 45°, m = 5    →  adds +5  to the 45° bin
```

**Result:** 64 pixels → **9 numbers** describing the local shape/texture.

---

### Step 4: Block Normalization (Lighting Invariance)

> **The Problem:** If lighting changes, all magnitudes scale proportionally — breaking the descriptor.

**The Fix:** Normalize over a **16×16 block** using the **L2-norm**:

$$\mathbf{v}_{\text{normalized}} = \frac{\mathbf{v}}{\sqrt{||\mathbf{v}||^2 + \epsilon}}$$

**Why it works:** If the image gets darker, both numerator and denominator shrink equally — the ratio stays constant. HOG becomes **robust to shadows and illumination changes**.

---

## Part 2: The Brain — SVM Classification

After HOG, we have a **feature vector** `x` (a long list of numbers). The SVM's job is to classify it: **Truck** or **Car**?

---

### Step 1: The Hyperplane (Decision Boundary)

The SVM finds a **hyperplane** — a flat boundary in high-dimensional space — that separates the two classes:

$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

| Symbol | Meaning |
|--------|---------|
| `w` | Weight vector (learned during training) |
| `x` | Input HOG feature vector |
| `b` | Bias term |

---

### Step 2: Maximizing the Margin

Many hyperplanes can separate two classes. SVM picks the **optimal** one — the hyperplane with the **maximum margin** (widest gap between classes).

The distance from the hyperplane to the nearest data point is:

$$\text{margin} = \frac{1}{||\mathbf{w}||}$$

**To maximize the margin, we minimize `||w||`.**

**Optimization Problem:**

$$\text{Minimize} \quad \frac{1}{2}||\mathbf{w}||^2$$

$$\text{Subject to:} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall\, i$$

> The constraint ensures every training sample is classified correctly and lies outside the margin boundary.

---

### Step 3: The Kernel Trick (Handling Non-Linearity)

Sometimes Trucks and Cars **cannot** be separated by a straight line (e.g. a small truck looks like a large car).

A **kernel function** `K(xᵢ, xⱼ)` implicitly maps data into a higher-dimensional space where a linear separator exists — **without ever computing the transformation explicitly**.

| Kernel | Use Case |
|--------|----------|
| **Linear** | Fastest; works when data is nearly linearly separable |
| **RBF (Gaussian)** | More accurate; handles complex, non-linear boundaries |

$$K_{\text{RBF}}(\mathbf{x}_i, \mathbf{x}_j) = \exp\!\left(-\gamma\,||\mathbf{x}_i - \mathbf{x}_j||^2\right)$$

---

## Elevator Pitch Summary

| Stage | Technical Description |
|-------|-----------------------|
| **Gradient Analysis** | Compute first-order derivatives `(Gx, Gy)` to capture edge intensity and orientation at every pixel |
| **Dimensionality Reduction** | Pool gradients into histograms within 8×8 cells to create a compact, robust shape descriptor |
| **L2-Normalization** | Normalize descriptor vectors over 16×16 blocks to ensure invariance to illumination changes |
| **Max-Margin Classification** | SVM maps HOG vectors into high-dimensional space and finds the optimal hyperplane `w·x + b = 0` maximizing the geometric margin between classes |

---

> **TL;DR:** HOG turns an image into a compact description of its edges and shapes. SVM draws the best possible decision boundary in that description space to tell Trucks from Cars.
