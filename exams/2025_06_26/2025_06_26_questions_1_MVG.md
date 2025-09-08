# MVG Classifiers
#### **Model Assumptions**

* Each class $C=c$ generates data according to a multivariate Gaussian distribution:

  $$
  X|C=c \sim \mathcal{N}(\mu_c, \Sigma_c)
  $$

  where:

  * $\mu_c$ is the class mean,
  * $\Sigma_c$ is the class covariance matrix.
* Samples are assumed to be i.i.d. (independent and identically distributed).
* Decision-making relies on Bayes’ theorem:

  $$
  P(C=c|x) \propto f(x|C=c) P(C=c)
  $$

---

#### **Estimation of Model Parameters**

* Parameters are estimated by Maximum Likelihood (ML) from the labeled training set:

  * **Mean:**

    $$
    {\mu}^*_c = \frac{1}{N_c} \sum_{i: y_i=c} x_i
    $$
  * **Covariance:**

    $$
    {\Sigma}^*_c = \frac{1}{N_c} \sum_{i: y_i=c} (x_i - {\mu}^*_c)(x_i - {\mu}^*_c)^T
    $$
  * $N_c$ = number of samples in class $c$.

---

#### **Inference / Classification**

* For a test sample $x$, compute the class posterior:

  $$
  P(C=c|x) = \frac{f(x|C=c) P(C=c)}{\sum_h f(x|C=h) P(C=h)}
  $$

* Decision rule: assign $x$ to the class with maximum posterior probability:

  $$
  \hat{c}(x) = \arg\max_c \; \log f(x|C=c) + \log P(C=c)
  $$

* **Multiclass**: directly choose the class maximizing the posterior.

* **Binary** ($C\in\{0,1\}$): use the log-likelihood ratio (LLR):

  $$
  \text{LLR}(x) = \log \frac{f(x|C=1)}{f(x|C=0)} + \log \frac{\pi_1}{\pi_0}
  $$

  Decision: choose class 1 if $\text{LLR}(x) > t$, else class 0.
  Threshold $t$ depends on prior $\pi$ and application costs.

* The decision boundaries:

  * **Unconstrained model**: quadratic (Quadratic Discriminant Analysis, QDA).
  * **Special cases**: linear (see below).

---

#### **Decision Rules (Binary Case)**

* General form:

  $$
  \hat{y}(x) = 
  \begin{cases}
  1 & \text{if } \log \frac{f(x|1)}{f(x|0)} > -\log \frac{\pi_1 C_{fn}}{(1-\pi_1) C_{fp}} \\
  0 & \text{otherwise}
  \end{cases}
  $$
* With equal priors and equal costs → threshold at 0.

---

#### **Naive Bayes Variant**

* **Assumption:** features are conditionally independent given the class.
  Covariance matrix becomes diagonal:

  $$
  \Sigma_c = \text{diag}(\sigma^2_{c,1}, \dots, \sigma^2_{c,D})
  $$
* **Decision rule:** still quadratic, but covariance simplified.
* **Benefits:** fewer parameters to estimate, works better with small sample sizes.
* **Limitations:** unrealistic independence assumption can harm performance if features are correlated.

---

#### **Tied Covariance Variant**

* **Assumption:** all classes share the same covariance:

  $$
  \Sigma_c = \Sigma \quad \forall c
  $$

  Estimated as the pooled covariance:

  $$
  \hat{\Sigma} = \frac{1}{N} \sum_c \sum_{i: y_i=c} (x_i - \hat{\mu}_c)(x_i - \hat{\mu}_c)^T
  $$
* **Decision rule:** log-likelihood ratio becomes linear in $x$ (Linear Discriminant Analysis, LDA).
* **Benefits:** robust covariance estimation when data is limited, reduced variance.
* **Limitations:** if real covariances differ significantly across classes, model is misspecified and performance degrades.

---

### **Summary**

* **MVG (unconstrained)**: flexible, quadratic boundaries, best when many samples are available.
* **Naive Bayes**: diagonal covariance, simpler and effective with few samples but ignores correlations.
* **Tied covariance**: linear boundaries, reliable with limited data if class covariances are similar, but less expressive if they are not.
