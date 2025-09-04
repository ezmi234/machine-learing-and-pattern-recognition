### **LDA vs. Tied MVG Classifier**

#### **LDA (binary case)**

* **Formulation & assumptions**:
  Linear Discriminant Analysis assumes each class distribution is Gaussian with *equal covariance matrix* Σ but different means μ₁, μ₂.
  It is a *discriminant approach*: it looks for the direction that best separates the classes.
* **Training objective**:
  Maximize the ratio of between-class to within-class variance:

  $$
  L(w) = \frac{w^T S_B w}{w^T S_W w}.
  $$

  For the binary case, the optimal direction is

  $$
  w \propto S_W^{-1} (\mu_2 - \mu_1).
  $$
* **Inference**:
  Project data x onto w, classify according to whether wᵀx is above or below a threshold t (usually midway between projected class means).
  Decision rule: **linear hyperplane** in the original space.

#### **Tied MVG classifier (binary case)**

* **Model formulation & assumptions**:
  This is a **generative model**. Each class is modeled as a **Multivariate Gaussian with its own mean** but a **tied covariance matrix**:

  $$
  X|C=c \sim \mathcal{N}(\mu_c, \Sigma), \quad \Sigma \text{ shared across classes}.
  $$

* **Training objective**:
  Maximum Likelihood estimates:

  $$
  \mu_c = \frac{1}{N_c} \sum_{i|c_i=c} x_i, \quad
  \Sigma = \frac{1}{N} \sum_{c} \sum_{i|c_i=c} (x_i - \mu_c)(x_i - \mu_c)^T
  $$

  .

* **Inference**:
  Compute the **log-likelihood ratio**:

  $$
  \text{llr}(x) = \log \frac{f(x|\mu_1,\Sigma)}{f(x|\mu_0,\Sigma)} + \log\frac{\pi}{1-\pi}.
  $$

  This simplifies to a **linear discriminant function** wᵀx + b.
  Decision rule: choose class 1 if llr(x) > threshold.

#### **Relationship between the two models**

* Both lead to **linear decision boundaries**.
* Under Gaussian assumptions with tied covariance, the Tied MVG decision rule coincides with LDA.


### **4. Decision Rules**

* **LDA binary**:
  Project onto discriminant vector $w$.
  Decision rule:

  $$
  \text{decide } C=1 \quad \text{if } w^T x > t
  $$

  where $t$ is a threshold (often midway between class projections).

* **Tied MVG binary**:
  Log-likelihood ratio test:

  $$
  \text{decide } h_1 \quad \text{if } w^T x + b > 0.
  $$

  Equivalent to LDA linear rule.

---

### **5. Multiclass LDA as Dimensionality Reduction**

* **Objective**: maximize class separability by finding a projection matrix $W$ that maximizes

  $$
  L = \text{Tr}((W^T S_W W)^{-1} (W^T S_B W)),
  $$

  where $S_W$ = within-class scatter, $S_B$ = between-class scatter.
  
  Solution: top *m* eigenvectors of $S_W^{-1}S_B$ with largest eigenvalues.

* **Limitations**:

  * At most $C-1$ meaningful projection directions (rank($S_B$) ≤ C-1).
  * Requires invertible $S_W$, which may fail in high dimensions (so PCA preprocessing often used).
  * Only linear separability captured; not effective if classes are not linearly separable.


---

**In summary**:

* **LDA**: discriminant, finds optimal projection maximizing class separation.
* **Tied MVG**: generative Gaussian model with tied covariance.
* Both lead to identical **linear decision rules** in the binary case.
* In the multiclass case, LDA is widely used as a **dimensionality reduction** method, though limited to C−1 directions and linear separation.


