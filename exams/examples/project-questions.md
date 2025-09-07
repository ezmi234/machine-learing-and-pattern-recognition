### Project - question example 1
Explain, in light of the characteristics of the classifiers and of the characteristics of the project
datasets:
1. The relative performance of the MVG, Tied MVG and GMM models.
2. The relative performance of linear and non-linear SVM.

### Answer
#### **1. Relative performance of the MVG, Tied MVG, and GMM models**

Based on the experimental results across Labs 5, 7, and 10, we observe that the **Gaussian Mixture Model (GMM) with diagonal covariance** delivers the best overall performance. Specifically, when modeling class 0 with 8 components and class 1 with 32 components, it achieves a **minimum Detection Cost Function (minDCF) of 0.1312** and an **actual DCF (actDCF) of 0.1517**. This significantly outperforms both the **standard Multivariate Gaussian (MVG)** and its variants:

* **MVG (full covariance)** achieves a comparable minDCF (\~0.1303) but lacks the flexibility to model multimodal class distributions.
* **Tied MVG**, which assumes a shared covariance matrix across all classes, shows consistently poorer performance, due to its inability to adapt to differences in class variances.
* **Naive Bayes MVG** performs surprisingly well, almost matching standard MVG, which is explained by the **low feature correlations** observed in the dataset.

The diagonal GMM performs better than the full covariance GMM because it avoids overfitting and is better aligned with the data structure (i.e., near-elliptical or circular clusters). Overall, **GMM models, especially with diagonal covariance, prove more expressive and robust**, particularly in datasets with class-conditional multimodalities.

#### **2. Relative performance of linear and non-linear SVM**

Support Vector Machines (SVM) with an RBF kernel demonstrate good discriminative power, achieving a **minDCF of 0.1773**, outperforming logistic regression (minDCF = 0.2436). However, **SVM suffers from severe miscalibration**, with actual DCFs peaking as high as **0.428**, especially at extreme prior log-odds. This behavior limits its practical reliability across applications with varying cost and prior configurations.

In contrast, **logistic regression (quadratic)** exhibits both poor minDCF and high actDCF, indicating suboptimal performance and calibration.

GMM, on the other hand, remains **consistently well-calibrated** across the entire operating range (log-odds from −4 to +4), with DCF curves closely matching the minimum DCF curves in the Bayes error plots. This makes it the **most reliable and effective classifier** for a wide range of applications.

---

### **Conclusion**

Considering both discrimination and calibration, the final ranking of classifiers is:

1. **GMM (diagonal covariance)** – Best performance and calibration.
2. **MVG (full covariance)** – Solid performance but limited flexibility.
3. **SVM (RBF kernel)** – Strong theoretical performance but poor calibration.
4. **Logistic Regression (quadratic)** – Worst overall performance.

**GMM with diagonal covariance emerges as the most promising method** for the given application (effective prior \$\tilde{\pi} = 0.1\$) and remains robust across various operating conditions.

---

### Project - question example 2
Explain the relative performance on the project validation set of diﬀerent SVM kernels (including
linear models), in light of the characteristics of the kernel and the characteristics of the dataset.
Briefly analyze the eﬀects of regularization on the model performance

### Answer

#### **Relative performance of different SVM kernels**

We evaluated the performance of three SVM kernel types—**Linear**, **Polynomial**, and **RBF (Radial Basis Function)**—on the project validation set, using both minimum Detection Cost Function (minDCF) and actual DCF (actDCF) as performance metrics under a target application with effective prior \$\tilde{\pi} = 0.1\$.

---

#### 🔹 **Linear SVM**

* The **minDCF curve is relatively flat**, with the best value achieved at **C = 0.1** (minDCF ≈ 0.358).
* However, **actDCF is strongly affected by regularization**: for small C (strong regularization), actDCF reaches **values close to 1**, indicating very poor performance.
* As C increases (weaker regularization), actDCF improves and stabilizes around **0.5**.
* **Centering the data** does not significantly alter these results.
* **Conclusion**: Linear SVM fails to capture the complexity of the dataset and suffers from **poor calibration** (large gap between minDCF and actDCF), which limits its practical utility.

---

#### 🔹 **Polynomial Kernel SVM (degree 2)**

* The model improves over the linear version: **minDCF drops to 0.245** (best at C ≈ 0.03), showing enhanced ability to capture **non-linear boundaries**.
* actDCF behavior mirrors the linear case: very high for low C, improves as C increases, reaching **\~0.4** at higher C.
* Again, **the model is miscalibrated**, but to a **lesser degree than linear SVM**.
* Performance is competitive with other quadratic models (e.g., Quadratic Logistic Regression).

---

#### 🔹 **RBF Kernel SVM**

* The **best performance overall** is achieved with:

  * \$\gamma = e^{-2}\$ and \$C = 31.6228\$
  * **minDCF = 0.1773**, **actDCF = 0.4276**
* This setup significantly outperforms both linear and polynomial SVMs in terms of minDCF.
* However, actDCF remains high (≈ 0.43), confirming persistent **calibration issues**.
* **RBF kernels are effective at modeling complex decision boundaries**, exploiting non-linearity in the feature space.

---

### **Effect of Regularization (C)**

Across all SVM variants:

* **Very small C values (strong regularization)** lead to **underfitting**: high minDCF and especially high actDCF.
* As C increases, **model flexibility improves**, leading to:

  * Stable or slightly improved minDCF
  * Strong reduction in actDCF (especially evident in RBF and polynomial kernels)
* The **regularization coefficient plays a crucial role** in balancing bias and variance, and is particularly important for **model calibration**.

---

### **Final Observations**

| Kernel         | Best minDCF | Best actDCF | Calibration | Notes                                        |
| -------------- | ----------- | ----------- | ----------- | -------------------------------------------- |
| **Linear**     | 0.358       | 0.49–1.0    | ❌ Poor      | Weak modeling capacity                       |
| **Poly (d=2)** | 0.245       | \~0.40      | ⚠️ Moderate | Better modeling, still miscalibrated         |
| **RBF**        | **0.177**   | 0.427       | ❌ Poor      | Best discriminative power, worst calibration |

* **RBF SVM is the most powerful discriminative model**, but its effectiveness is **limited by calibration**.
* **Polynomial kernel SVMs** offer a good trade-off between performance and complexity.
* **Linear SVM** is too rigid for the characteristics of the dataset and performs poorly even after tuning.

---

### Project - question example 3
Consider the SVM and logistic regression classifiers. In lights of the characteristics of the datasets
and of the classifiers, explain the gap between minimum and actual DCF for each model, and,
if necessary, the method that you employed to reduce this gap for the project dataset. Analyze
also the eﬀects of regularization on the miscalibration error for both models.

### Answer

#### **Gap between Minimum and Actual DCF**

Both **Support Vector Machines (SVM)** and **Logistic Regression (LR)** classifiers exhibit a notable gap between **minimum DCF (minDCF)** and **actual DCF (actDCF)** across various configurations. This gap is indicative of **miscalibration**, meaning that while the models are capable of good discrimination (as minDCF is low), the scores they produce **do not reflect true posterior probabilities**, thus harming decision-making performance in real applications.

---

#### 🔹 **Support Vector Machines (SVM)**

* SVMs are inherently **margin-based classifiers**: they focus on finding a hyperplane that maximizes the separation between classes but **do not optimize probabilistic outputs**.
* As a result, **their raw decision scores are uncalibrated**, especially under extreme prior conditions (e.g., \$\tilde{\pi} = 0.1\$), leading to **large actDCF values** (e.g., up to **0.43** even with RBF kernels).
* This miscalibration is visible across all kernel types—linear, polynomial, and RBF—and **becomes worse with stronger regularization** (low values of C), which can **compress the score distribution** and distort the scale.
* **Calibration methods** (e.g., Platt scaling or isotonic regression) could be used to reduce this gap, but were not explicitly employed in this project.

---

#### 🔹 **Logistic Regression**

* Logistic regression naturally outputs **posterior probabilities** via the sigmoid function, and is **theoretically well-calibrated** under ideal conditions.
* Despite this, a gap between minDCF and actDCF is still observed in practice, especially under:

  * **High regularization** (large \$\lambda\$): causes **underfitting**, making the predicted probabilities too flat and uncertain.
  * **Low regularization** (small \$\lambda\$): can lead to **overfitting**, where the model becomes too confident in noisy or specific patterns.
* In our experiments, even the best configuration (quadratic LR with \$\lambda = 0.0316\$) showed a gap (minDCF = 0.2436, actDCF = 0.4972), which suggests that **regularization directly impacts calibration quality**.

---

### **Effect of Regularization on Miscalibration**

| Model Type        | Low Regularization (High C / Low λ) | High Regularization (Low C / High λ) |
| ----------------- | ----------------------------------- | ------------------------------------ |
| **SVM**           | Better actDCF, still uncalibrated   | Very poor actDCF (≈1), large gap     |
| **Logistic Reg.** | Better calibration if λ is optimal  | Underfitting → flat probabilities    |

* In both models, **too much regularization increases the gap** due to reduced model capacity.
* The **sweet spot** for each model needs to be tuned via validation to **minimize the miscalibration error**.

---

### **Conclusion**

* The gap between minDCF and actDCF is **inherent to the model formulation** and **exacerbated by poor regularization choices**.
* **SVM** suffers from **structural miscalibration** due to its non-probabilistic nature.
* **Logistic regression**, while better in theory, still requires **careful tuning of regularization** to preserve calibration.
* Although no explicit calibration methods (e.g., score transformation) were used in this project, future work could apply **post-hoc calibration techniques** to align scores with actual class probabilities and improve actDCF performance.

---

### Project - question example 4
Given the following functions (assume these functions are already implemented unless specified):
- **trainPCA**: trains a PCA model
- **applyPCA**: applies a PCA model to some data
- **trainClassifier(D, L)**: trains a given classifier from the data matrix D and the label
vector L; returns an object containing the trained model parameters
- **scoreClassifier(clsModel, D)**: computes the array of scores for classifier clsModel (as
returned by the function trainClassifier) for the samples in data matrix D
- **evaluateScores(S, L)**: computes a performance metric (e.g. minimum DCF) over the
score array S with label vector L
a) Provide a possible signature and an implementation of the functions trainPCA and applyPCA,
briefly explaining also the function parameters and the return value.
b) Using these functions, write the Python code to:
- Train the classifier on a training set, optimizing the PCA dimension with respect to the
provided metric function using a single-fold cross-validation approach
- Evaluate its performance on an evaluation set.
Assume that you have at your disposal a training set, already divided in model training data
(DTR, LTR) and validation data (DVAL, LVAL), and an evaluation set (DTE, LTE). DTR, DVAL
and DTE are data matrices, with samples organized as column vectors, whereas LTR, LVAL and
LTE are arrays containing the corresponding labels. To select the PCA dimension m consider all
possible values of mthat are compatible with the dimension of the feature vectors. Assume that
the classifier is invariant to aﬃne transformations, that it does not include hyper-parameters to
tune, and that PCA is the only kind of pre-processing to analyze.
Summary of main numpy (np) and scipy employed in the laboratories:
- **s, U = np.linalg.eigh\(C\)**:
returns the array of eigenvalues s in ascending order and the matrix of corresponding eigenvectors U of a real symmetric matrix C.
- **U, s, Vh = np.linalg.svd\(C\)**: returns the array of singular values s in descending order,the correspondin g matrix of left singular vectors U and the corresponding transposed matrix of right singular vectors Vh.
- **s, v = np.linalg.slogdet\(C\)**: returns the sign s and the logarithm of the absolute value v of the determinant of matrix C
- **v = scipy.special.logsumexp(M, axis=k)**: computes in a numerically more stable way np.log(np.sum(np.exp(a),axis=k))
- **v = np.logaddexp(a, b)**: computes in a numerically more stable way np.log(np.exp(a) + np.exp(b))

### Answer

#### **a) Signature and implementation of `trainPCA` and `applyPCA`**

---

##### **Function Signatures**

```python
def trainPCA(D: np.ndarray, m: int) -> np.ndarray:
    """
    Trains a PCA model to reduce the data to m dimensions.

    Parameters:
    - D: numpy.ndarray of shape (n_features, n_samples), input data matrix
    - m: int, number of principal components to retain

    Returns:
    - P: numpy.ndarray of shape (m, n_features), projection matrix for PCA
    """
    ...

def applyPCA(P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Applies the PCA projection to reduce data to m dimensions.

    Parameters:
    - P: numpy.ndarray of shape (m, n_features), PCA projection matrix
    - D: numpy.ndarray of shape (n_features, n_samples), input data matrix

    Returns:
    - D_reduced: numpy.ndarray of shape (m, n_samples), projected data
    """
    ...
```

---

##### **Function Implementations**

```python
def trainPCA(D: np.ndarray, m: int) -> np.ndarray:
    # Compute the empirical mean
    mu = D.mean(axis=1, keepdims=True)

    # Center the data
    DC = D - mu

    # Compute the covariance matrix
    C = np.dot(DC, DC.T) / D.shape[1]

    # Compute eigenvalues and eigenvectors
    s, U = np.linalg.eigh(C)  # ascending order

    # Take the m largest eigenvectors (in descending order)
    P = U[:, ::-1][:, :m]  # flip and select m columns

    return P.T  # Return projection matrix of shape (m, n_features)

def applyPCA(P: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.dot(P, D)  # Project data using the projection matrix
```

---

#### **b) Code for PCA selection and evaluation**

```python
import numpy as np

def optimize_and_evaluate(DTR, LTR, DVAL, LVAL, DTE, LTE, evaluateScores):
    """
    Trains a classifier using PCA-preprocessed data, optimizes PCA dimension on validation set,
    and evaluates final performance on evaluation set.

    Parameters:
    - DTR, LTR: training data and labels
    - DVAL, LVAL: validation data and labels
    - DTE, LTE: evaluation data and labels
    - evaluateScores: function to compute performance (e.g., minDCF)

    Returns:
    - best_m: optimal PCA dimension
    - best_metric: performance score on validation set
    - eval_metric: performance score on evaluation set using best PCA
    """

    n_features = DTR.shape[0]
    best_metric = float('inf')
    best_m = None
    best_model = None
    best_P = None

    for m in range(1, n_features + 1):
        # Train PCA
        P = trainPCA(DTR, m)

        # Apply PCA to training and validation sets
        DTR_PCA = applyPCA(P, DTR)
        DVAL_PCA = applyPCA(P, DVAL)

        # Train classifier
        model = trainClassifier(DTR_PCA, LTR)

        # Get validation scores
        SVAL = scoreClassifier(model, DVAL_PCA)

        # Evaluate performance
        score = evaluateScores(SVAL, LVAL)

        if score < best_metric:
            best_metric = score
            best_m = m
            best_model = model
            best_P = P

    # Apply PCA to evaluation set using best P
    DTE_PCA = applyPCA(best_P, DTE)

    # Evaluate on test set
    STE = scoreClassifier(best_model, DTE_PCA)
    eval_metric = evaluateScores(STE, LTE)

    return best_m, best_metric, eval_metric
```

---

### **Explanation**

* **trainPCA** computes a projection matrix by extracting the top `m` eigenvectors of the data covariance matrix.
* **applyPCA** projects data onto the reduced-dimensional space.
* The **optimization loop** uses a simple single-fold cross-validation (DTR vs DVAL) to find the best number of PCA components.
* The final model is trained on the **PCA-transformed training data** and tested on the **evaluation set** using the same PCA projection.

---
Project - question example 5
You are given the following functions (assume these functions are already implemented unless
specified):

- trainRBFKernelSVM(D, L, C, gamma): trains an SVM model with an RBF kernel with
    hyper-parametergammaand returns an object containing the trained model information;D
    is the training data matrix,Lis the corresponding label array, andCis the SVM cost-vs-
    margin trade-off coefficient
- scoreRBFKernelSVM(svmModel, D): computes the classification scores for samples in
    the data matrix D for an SVM model svmModel (as returned by the function
    trainRBFKernelSVM) and returns an array of scores
- evaluateScores(S, L): computes an evaluation metric (e.g. minimum DCF) over the
    array of scoresSwith associated array of labelsL

Assume that you have at your disposal a training set, already divided in model training data
(DTR, LTR)and validation data(DVAL, LVAL), and an evaluation set(DTE, LTE).DTR,DVAL
andDTE are data matrices, with samples organized as column vectors, whereasLTR,LVALand
LTEare arrays containing the corresponding labels.

Write the Python code to train and apply an SVM classifier. In particular, the code should

- Train an SVM classifier, optimizing the value of the hyper-parameters with respect to the
    metric functionevaluateScoresusing a single-fold cross-validation approach.
- Evaluate the selected SVM model on the evaluation data, using the provided metric.

Write an implementation of scoreRBFKernelSVM(svmModel, D). Assume thatsvmModelis an
object with the following fields:

```
sv: numpy2-D array of support vectors, stored as column vectors
```
```
alpha: Lagrange multiplier values associated to each sv, as a 1-Dnumpyarray
```
```
labels: 1-Dnumpyarray of labels (+1 or -1) associated to the support vector
```
```
gamma: RBF kernel hyper-parameterγ
```
You can assume that you have at your disposal a functionRBFKernel(D1, D2, gamma)that
returns the matrixof kernel valuesk(x,y)for all pairs of samplesx,yof 2-D sample matricesD1,
D2(i.e., ifK = RBFKernel(D1, D2, gamma), thenK[i, j]is the kernel between arraysD1[:,
i]andD2[:, j]).

### Answer

#### ✅ **Step 1: Implement `scoreRBFKernelSVM`**

```python
def scoreRBFKernelSVM(svmModel, D):
    """
    Computes the SVM scores for the input data matrix D using the RBF kernel SVM model.

    Parameters:
    - svmModel: object with fields:
        - sv: support vectors (n_features, n_support_vectors)
        - alpha: alpha coefficients (n_support_vectors,)
        - labels: labels of support vectors (+1 or -1) (n_support_vectors,)
        - gamma: RBF kernel parameter
    - D: data matrix (n_features, n_samples)

    Returns:
    - scores: numpy array of shape (n_samples,), score for each input sample
    """
    K = RBFKernel(svmModel.sv, D, svmModel.gamma)  # shape: (n_support_vectors, n_samples)
    weighted_labels = svmModel.alpha * svmModel.labels  # element-wise multiplication
    scores = np.dot(weighted_labels, K)  # (n_samples,)
    return scores
```

---

#### ✅ **Step 2: Training with hyperparameter tuning (γ and C) and evaluating on test set**

```python
def optimize_and_evaluate_SVM(DTR, LTR, DVAL, LVAL, DTE, LTE, evaluateScores):
    """
    Trains an RBF Kernel SVM, optimizing gamma and C with respect to evaluateScores on validation set,
    and evaluates final performance on test set.

    Returns:
    - best_gamma, best_C: optimal hyperparameters
    - best_val_score: score on validation set
    - test_score: score on evaluation set
    """
    import numpy as np

    gammas = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]
    Cs = np.logspace(-3, 2, 11)

    best_score = float('inf')
    best_model = None
    best_gamma = None
    best_C = None

    for gamma in gammas:
        for C in Cs:
            # Train model
            model = trainRBFKernelSVM(DTR, LTR, C, gamma)

            # Score validation set
            SVAL = scoreRBFKernelSVM(model, DVAL)

            # Evaluate
            score = evaluateScores(SVAL, LVAL)

            if score < best_score:
                best_score = score
                best_model = model
                best_gamma = gamma
                best_C = C

    # Evaluate on test set
    STE = scoreRBFKernelSVM(best_model, DTE)
    test_score = evaluateScores(STE, LTE)

    return best_gamma, best_C, best_score, test_score
```

---

### **Explanation**

* `scoreRBFKernelSVM` implements the dual form of the SVM decision function using only the **support vectors** and their associated **α, labels**, and **RBF kernel**.
* The `optimize_and_evaluate_SVM` function performs a **grid search** over combinations of `C` and `γ`, selecting the model with the best performance on the validation set (`minDCF`, or other metrics).
* Final performance is computed on the **held-out evaluation set**.

---

### **Conclusion**

This approach ensures that the chosen SVM model generalizes well by:

* Evaluating performance using a proper validation split.
* Avoiding overfitting by selecting hyperparameters based on actual metric values.
* Applying a correct SVM scoring function that uses kernel similarities and dual coefficients.

---

### Project - question example 6

Consider a binary classification problem, with classes labeled as 1 and 0, respectively.

Let `(DTR, LTR)`, `(DVAL, LVAL)` represent a labeled training set and a labeled validation

set. `DTR` and `DVAL` are 2-D numpy arrays containing the dataset samples (stored as column
vectors), whereas `LTR` and `LVAL` are 1-D numpy arrays containing the sample labels. Let also
`DTE` represent the dataset matrix (again, a 2-D numpy array) containing the samples that our
application should classify.

Write a Python code fragment that:

1. trains a calibrated binary classifier
2. performs inference (i.e. computes predicted labels) on the evaluation data

You can assume that the following functions have been defined:

- `trainClassifier(D, L)`: train a non-calibrated classification model (e.g., an SVM or
    an LDA classifier) on the training matrix `D` with associated labels array `L`, and return
    a python object containing the trained model (assume that the model does not contain
    tunable hyper-parameters)
- `scoreClassifier(model, D)`: compute the <u>non-calibrated</u> classification scores for model
    `model` (as returned by `trainClassifier`) for the samples in data matrix `D` and return
    a 1-D array of scores
- `trainCalibrationModel(S, L, prior)`: train a calibration model on the 1-D array of
    scores S, with associated array of labels L, for a binary application with prior `prior` for
    class 1, and return a python object containing the trained model
- `applyCalibrationModel(calModel, S)`: apply the calibration model `calModel` (as re-
    turned by `trainCalibrationModel`) to the 1-D array of scores `S`, and return a 1-D array
    of calibrated scores

NOTE: assume that the target application is characterized by an effective prior `p` for class 1.

You are <u>not required</u> to tune the calibration model hyper-parameter `prior`, but you can assume
that the calibration model can be trained using the target application prior `p`.

### Answer
#### **Objective:**

Train a **calibrated binary classifier** and use it to compute **predicted labels** on an **evaluation dataset**, assuming:

* A **prior** probability `p` for class 1 is known.
* Functions to train a classifier, score it, calibrate scores, and apply calibration are already defined.

---

### ✅ **Python Code**

```python
# Step 1: Train the base (non-calibrated) classifier on training data
model = trainClassifier(DTR, LTR)

# Step 2: Score the validation set using the non-calibrated model
SVAL = scoreClassifier(model, DVAL)

# Step 3: Train a calibration model using validation scores and labels
calModel = trainCalibrationModel(SVAL, LVAL, prior=p)

# Step 4: Score the evaluation set (non-calibrated)
STE_uncalibrated = scoreClassifier(model, DTE)

# Step 5: Apply the calibration model to the scores
STE_calibrated = applyCalibrationModel(calModel, STE_uncalibrated)

# Step 6: Convert calibrated scores to predicted labels using the effective threshold
# Optimal Bayes threshold: log(p / (1 - p))
import numpy as np
threshold = -np.log(p / (1 - p))

predicted_labels = (STE_calibrated > threshold).astype(int)
```

---

### ✅ **Explanation**

* **`trainClassifier`**: learns a binary classifier on the training set (`DTR`, `LTR`).

* **`scoreClassifier`**: returns uncalibrated log-likelihood ratio (LLR) scores.

* **`trainCalibrationModel`**: adjusts the LLRs to reflect the actual posterior probability distribution (e.g., via logistic regression).

* **`applyCalibrationModel`**: maps raw scores to calibrated scores.

* The decision threshold is computed from the **Bayes decision rule**:

  $$
  \text{threshold} = -\log\left(\frac{p}{1 - p}\right)
  $$

  where $p$ is the effective prior probability of class 1.

* **Final prediction** is obtained by thresholding the calibrated scores.

---

### ✅ **Conclusion**

This code pipeline ensures:

* Proper separation between **training** and **calibration** sets to avoid overfitting.
* Correct **use of prior** in the calibration model and decision threshold.
* Robust and interpretable output labels from a **calibrated probabilistic classifier**.


