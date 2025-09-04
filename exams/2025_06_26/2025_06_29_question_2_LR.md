### Binary logistic regression

The binary logistic regression is a discriminative probabilistic model with linear decision boundaries

**Classification rule**:
- Model computes a **score**:
  $$
  s(x)=w^tx+b 
  $$
- Classification rule: assign class 1 if $s(x)\ge t$, class 0 otherwise.
- With equal priors and symmetric costs, the optimal threshold is $t=0$.
- Decision boundaries are linear hyperplanes orthogonal to $w$.

**Probabilistic interpretation** 
- Logistic regression directly models posterior probability:
  $$
  P(C=1|x) = \sigma(s(x)) = \frac{1}{1 + e^{-s(x)}}, \quad
  P(C=0|x) = 1 - \sigma(s(x)).
  $$
  * The score $s(x)$ corresponds to the **log-posterior odds**:

  $$
  s(x) = \log \frac{P(C=1|x)}{P(C=0|x)}.
  $$
* Interpretation: smooth probabilistic mapping from linear function of x.

---

#### **Parameter estimation & training objective**

* Training data: $\{(x_i, c_i)\}_{i=1}^N$, with $c_i \in \{0,1\}$.
* Likelihood:

  $$
  L(w,b) = \prod_{i=1}^N P(C=c_i|x_i,w,b).
  $$
* Log-likelihood:

  $$
  \ell(w,b) = \sum_{i=1}^N \big[c_i \log \sigma(s_i) + (1-c_i)\log(1-\sigma(s_i))\big].
  $$
* Training = **maximize log-likelihood**, equivalently **minimize cross-entropy loss**:

  $$
  J(w,b) = -\ell(w,b).
  $$
* Alternative view: **risk minimization** with logistic loss

  $$
  \ell(z,s) = \log(1 + e^{-z s}), \quad z \in \{-1,+1\}.
  $$
* Parameters are estimated numerically (no closed form), usually via gradient descent or Newton methods.
* Regularization (e.g., $\lambda \|w\|^2/2$) is often added to prevent overfitting.

---

#### **Non-linear classification extension**

* Replace input features $x$ with a **non-linear mapping** $\phi(x)$.
* Model becomes:

  $$
  s(x) = w^T \phi(x) + b.
  $$
* The decision boundary is linear in $\phi(x)$ but non-linear in original space.
* Example: polynomial feature expansion, kernelized logistic regression.

---

#### **Extension to score calibration**

* Logistic regression scores already approximate **log-likelihood ratios (LLRs)**, but can be **biased or miscalibrated** if regularization or prior mismatch occurs.
* **Calibration approaches**:

  * **Prior-weighted logistic regression**: train with effective prior $\tilde{\pi}$ reflecting application needs.
  * **Affine calibration**: post-process raw scores with

    $$
    s_{cal}(x) = \alpha s(x) + \beta,
    $$

    where $\alpha,\beta$ are fitted on a calibration set via logistic regression.
* This provides **well-calibrated LLRs** usable across applications with different priors and costs.
