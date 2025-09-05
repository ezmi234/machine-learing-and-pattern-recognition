### Binary logistic regression

The binary logistic regression is a discriminative probabilistic model with linear decision boundaries. In particular, for binary case it aims at modeling the class posterior probabilities.
$$
P(C=1|x), \ P(C=0|x)=1-P(C=1|x)
$$
where $x$ is the sample that we want to classify, and the possible class labels are 1 and 0.

**Classification rule**:
- The model assumes that the separation surface between the classes
is a linear surface (hyperplane), corresponding to linear decision
functions of the form:
  $$
  s(x; w, b)=w^tx+b  \lessgtr t
  $$
- $(w,b)$ are the model parameters and represent the separation
surface.
- Classification rule: assign class 1 if $s(x)\ge t$, class 0 otherwise.
- With equal priors and symmetric costs, the optimal threshold is $t=0$.
- Decision boundaries are linear hyperplanes orthogonal to $w$.

**Probabilistic interpretation** 
* The score $s(x)$ corresponds to the **log-posterior odds**:

  $$
  s(x) = \log \frac{P(C=1|x)}{P(C=0|x)}.
  $$
- So, posterior probabilities for the classes are computed as: 
  $$
  P(C=1|x; w, b) = \sigma(s(x; w,b)) = \frac{1}{1 + e^{-s(x)}}, \\ P(C=0|x; w,b) = 1 - \sigma(s(x;w,b)).
  $$
- Where $\sigma$ denotes the sigmoid function:
![sigmoid](./sigmoid.png) 
* Interpretation: smooth probabilistic mapping from linear function of x.

---

#### **Parameter estimation & training objective**

* Each label $c_i \in \{0,1\}$ is modeled as a **Bernoulli random variable** with parameter $\sigma(s(x_i))$:

  $$
  P(C=c_i|x_i,w,b) = \sigma(s_i)^{c_i}(1-\sigma(s_i))^{1-c_i}.
  $$
* Likelihood (i.i.d. samples):

  $$
  L(w,b) = \prod_{i=1}^N P(C=c_i|x_i,w,b).
  $$
* Log-likelihood:

  $$
  \ell(w,b) = \sum_{i=1}^N \big[c_i\log\sigma(s_i) + (1-c_i)\log(1-\sigma(s_i))\big].
  $$
* Training = **maximize log-likelihood** or equivalently **minimize cross-entropy**:

  $$
  J(w,b) = -\ell(w,b).
  $$
* Alternative view: **risk minimization** with logistic loss

  $$
  \ell(z,s) = \log(1+e^{-z s}), \quad z \in \{-1,+1\}.
  $$
* Optimization is numerical (no closed form). Regularization (e.g. $\tfrac{\lambda}{2}\|w\|^2$) controls complexity and avoids overfitting .

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
