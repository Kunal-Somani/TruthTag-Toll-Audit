# tiet-ucs532p-bteam

## 📐 Mathematical Framework

This project eschews "black-box" Deep Learning in favor of a transparent, interpretable Classical Computer Vision pipeline. The core methodology relies on **Gradient-Based Feature Extraction** coupled with **Max-Margin Statistical Classification**.

### 1. Feature Extraction: Histogram of Oriented Gradients (HOG)

We transform raw pixel data into a structural feature vector $\mathbf{v}$ that is invariant to illumination and small geometric deformations.

#### **A. Gradient Computation**
For every pixel $(x,y)$, we calculate the gradient vector $\nabla I(x,y)$ to capture edge intensity and direction. We use discrete derivative masks $[-1, 0, 1]$ centered at the pixel:

$$
G_x(x,y) = I(x+1, y) - I(x-1, y)
$$

$$
G_y(x,y) = I(x, y+1) - I(x, y-1)
$$

From these partial derivatives, we compute the **Magnitude** $m$ and **Orientation** $\theta$:

$$
m(x,y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2}
$$

$$
\theta(x,y) = \arctan\left(\frac{G_y(x,y)}{G_x(x,y)}\right)
$$

#### **B. Spatial Binning & Normalization**
To reduce noise and dimensionality, we aggregate gradients into $8 \times 8$ pixel **cells**. Each cell accumulates a weighted vote into a 9-bin histogram based on $\theta$.

To ensure robustness against lighting variations (e.g., shadows vs. sunlight), we normalize the vector $\mathbf{v}$ over larger $16 \times 16$ **blocks** using the **L2-Hys (Hysteresis) Norm**:

$$
\mathbf{v}_{norm} = \frac{\mathbf{v}}{\sqrt{||\mathbf{v}||_2^2 + \epsilon}}
$$

This results in a high-dimensional feature descriptor that represents the "shape fingerprint" of the vehicle.

---

### 2. Classification: Support Vector Machine (SVM)

The extracted HOG vectors are projected into a high-dimensional space where we construct an optimal separating hyperplane between classes (e.g., *Car* vs. *Truck*).

#### **A. The Optimization Problem**
We employ a **Linear SVM** to find the weight vector $\mathbf{w}$ and bias $b$ that maximizes the geometric margin $\gamma$ between the nearest data points (Support Vectors) and the decision boundary.

$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

Mathematically, we solve the following constrained convex optimization problem:

$$
\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^{N} \xi_i
$$

**Subject to:**
$$
y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

Where:
* $\frac{1}{2} ||\mathbf{w}||^2$: Maximizes the margin (regularization).
* $C$: Hyperparameter controlling the penalty for misclassification.
* $\xi_i$: Slack variables allowing for non-linearly separable data (soft margin).

---

### 3. Localization: Sliding Window & Non-Maximum Suppression (NMS)

To localize vehicles in the video feed, we scan the image $I$ with a fixed-size window $W$ at multiple scales.

For a set of detected bounding boxes $B = \{b_1, b_2, ..., b_n\}$ with confidence scores $S$, we eliminate redundant overlapping boxes using **Intersection over Union (IoU)**:

$$
IoU(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}
$$

We discard any box $b_j$ if $IoU(b_{best}, b_j) > \tau$ (where $\tau$ is the overlap threshold, typically 0.3), retaining only the most confident detection.
