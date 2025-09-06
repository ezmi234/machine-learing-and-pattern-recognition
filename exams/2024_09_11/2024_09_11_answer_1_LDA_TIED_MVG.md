### **LDA vs. Tied MVG Classifier**

#### **LDA (binary case)**

* **Formulation & assumptions**:

  Linear Discriminant Analysis (LDA) is a supervised linear method which seeks direction that best separate the classes.
  A direction is a unit vector $w$, its projection is $y=w^Tx$. The goal is to maximize the separation between classes
  and minimize the spread within classes:
  $$
  w^* = \max_w \frac{w^T S_B w}{w^T S_W w}.
  $$
  where:
  $$
  S_B = \frac{1}{N}\sum_{c=i}^k n_c(\mu_c-\mu)(\mu_c-\mu)^T
  $$
  $$
  S_W = \frac{1}{N}\sum_{c=i}^k \sum_{i=i}^{n_c} (x_{c,i}-\mu_c)(x_{c,i}-\mu_c)^T
  $$

  $x_{c,i}=\text{i-th}$ sample of class $c$; $n_c=$ # samples in class $c$; $k$ # of classes (0,1); $N$ # total number
  of samples; $\mu_c$ mean of class $c$; $\mu$ dataset mean.

  For the binary case, we solve $S_W^{-1}S_Bw=\lambda(w)w$, where the optimal choice is:

  $$
  w \propto S_W^{-1} (\mu_2 - \mu_1).
  $$
  So that $w$ is a line connecting the 2 means and we balanced using the within class matrix.
* **Training objective**:
  Estimate class means $\mu_1$ and $\mu_2$ and within class scatter $S_W$ from the training set, then compute $w$.
* **Inference**:
  Project data x onto w, classify according to whether wᵀx is above or below a threshold t (usually midway between
  projected class means).

* **Decision rule**: **linear hyperplane** in the original space.

#### **Tied MVG classifier (binary case)**

* **Model formulation & assumptions**:
  This is a **generative model**. Each class is modeled as a **Multivariate Gaussian with its own mean** but a **tied
  covariance matrix**, assuming data $x \in R^d$, indipendent and identically distributed and for each class $c$:

  $$
  X|C=c \sim \mathcal{N}(\mu_c, \Sigma), \quad \Sigma \text{ shared across classes}.
  $$

* **Training objective**:
  Estimate means and shared covariance through Maximum Likelihood:

  $$
  \mu_c = \frac{1}{N_c} \sum_{i|c_i=c} x_i, \quad
  \Sigma = \frac{1}{N} \sum_{c} \sum_{i|c_i=c} (x_i - \mu_c)(x_i - \mu_c)^T
  $$


Here’s a more cohesive and verbose rewrite of that **inference** part for the **Tied MVG classifier**, explicitly showing the log-likelihood ratio expansion, the meaning of the bias term $b$, and its relation to priors:

**Inference:**

For binary classification with classes $C \in \{0,1\}$, the decision is based on the **log-posterior ratio**:

$$
\log \frac{P(C=1|x)}{P(C=0|x)} \;=\; \log \frac{f(x|\mu_1,\Sigma)}{f(x|\mu_0,\Sigma)} + \log \frac{\pi}{1-\pi},
$$

where:

* $f(x|\mu_c,\Sigma)$ is the Gaussian density for class $c$,
* $\pi$ is the prior probability of class $c=1$.

The first term is the **log-likelihood ratio (llr)**, and the second term incorporates prior knowledge as a **log-prior odds**.

The **log-likelihood ratio (llr)** simplifies to a linear discriminant function wᵀx + b.

$$
\text{llr}(x) \;=\; w^T x + b,
$$

#### **Relationship between the two models**

* Both lead to **linear decision boundaries**.
* Under Gaussian assumptions with tied covariance, the Tied MVG decision rule coincides with LDA.
* The main difference is that LDA typically sets an **arbitrary** threshold, while the Tied MVG classifier places the threshold **according to the prior probabilities**.


### **4. Decision Rules**

* **LDA binary**:
  Project onto discriminant vector $w$.
  Decision rule:

  $$
  \text{decide } C=1 \quad \text{if } w^T x > t 
  $$
  $$
  \text{decide } C=0 \ \ \text{otherwise } 
  $$

  where $t$ is a threshold (often midway between class projections).

* **Tied MVG binary**:
  Log-likelihood ratio test:

  $$
  \text{decide } h_1 \quad \text{if } w^T x + b > t.
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
* In the multiclass case, LDA is widely used as a **dimensionality reduction** method, though limited to C−1 directions
  and linear separation.


