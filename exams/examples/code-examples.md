### Project - question 2 of exam 2025_06_26 (3 pt)

Given the following functions (assume these functions are already implemented):
- `trainPCA`: trains a PCA model
- `applyPCA`: applies a PCA model to some data
- `trainLDA`: trains a LDA model
- `applyLDA`: applies a LDA model to some data
- `evaluateScores(S, L)`: computes a performance metric (e.g. minimum DCF for a predefined application) over the score array S with label L
1. Provide possible signatures (prototype) for the `trainPCA`, `applyPCA`, `trainLDA`, `applyLDA` functions, briefly explaining the function parameters and the return value.
2. Using these functions, write a short Python program to train and apply an LDA binary classifier. The classifier should simply output a classification <u>score</u> (i.e. you are <u>not</u> required to provide predicted labels or select a suitable score threshold). The performance of the classifier can be evaluated through the provided function `evaluateScores`. Assume that you have at your disposal a training set, already directed in training data (DTR, LTR) and validation data (DVAL, LVAL) and an evaluation set (DTE, LTE). DTR, DVAL and DTE are data matrices, with samples organised as columns vectors, whereas LTR, LVAL and LTE are arrays containing the corresponding labels. The program should employ in an appropriate way the provided data to train the classification model and to select an optimal value for the PCA dimensionality scores and the corresponding performance for the evaluation set.

### Answer

```python
def trainPCA(D: np.ndarray, m: int) -> np.ndarray:
    """
    Train a PCA projection on data.
    Args:
        D: (d, n) data matrix with column-wise samples.
        m: target dimensionality (m <= d).
    Returns:
        P: (d, m) PCA projection matrix (columns are the top-m PCs).
    """
```

```python
def applyPCA(P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Apply a trained PCA projection.
    Args:
        P: (d, m) PCA projection matrix.
        D: (d, n) data matrix.
    Returns:
        DP: (m, n) projected data.
    """
```

```python
def trainLDA(D: np.ndarray, L: np.ndarray, m: int = 1) -> np.ndarray:
    """
    Train (Fisher) LDA directions.
    Args:
        D: (d, n) training data, column-wise samples.
        L: (n,) integer labels.
        m: number of LDA directions to keep (1 for binary).
    Returns:
        U: (d, m) LDA projection matrix.
    """
```

```python
def applyLDA(U: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Apply LDA projection.
    Args:
        U: (d, m) LDA projection matrix.
        D: (d, n) data matrix.
    Returns:
        S: (m, n) projected data; with m=1 this is the score row.
    """
```

### Step 2: Python program to train and apply an LDA binary classifier

```python
def lda_with_pca_pipeline(DTR, LTR, DVAL, LVAL, DTE, LTE, evaluateScores):
    """
    Train + evaluate an LDA binary classifier with PCA dimensionality selection.
    Returns test scores and the corresponding evaluation metric.
    """
    
    best_m, best_metric = None, np.inf
    for m in range(1, DTR.shape[0] + 1):
        P = trainPCA(DTR, m)
        DTR_pca, DVAL_pca = applyPCA(P, DTR), applyPCA(P, DVAL)

        U = trainLDA(DTR_pca, LTR, m=1)
        S_val = applyLDA(U, DVAL_pca).ravel()

        metric = evaluateScores(S_val, LVAL)
        if metric < best_metric:
            best_metric, best_m = metric, m

    # --- Step 2: retrain with best m on train+val, evaluate on test ---
    DTRVAL, LTRVAL = np.hstack([DTR, DVAL]), np.hstack([LTR, LVAL])
    P = trainPCA(DTRVAL, best_m)
    DTRVAL_pca, DTE_pca = applyPCA(P, DTRVAL), applyPCA(P, DTE)

    U = trainLDA(DTRVAL_pca, LTRVAL, m=1)
    S_test = applyLDA(U, DTE_pca).ravel()
    test_metric = evaluateScores(S_test, LTE)

    print(f"Best PCA m: {best_m}, Validation metric: {best_metric:.4f}")
    print(f"Test metric with m={best_m}: {test_metric:.4f}")
    return S_test, test_metric
```


# Project - question example 5
You are given the following functions (assume these functions are already implemented unless
specified):

- trainRBFKernelSVM(D, L, C, gamma): trains an SVM model with an RBF kernel with
    hyper-parameter gamma and returns an object containing the trained model information;D
    is the training data matrix,Lis the corresponding label array, and C is the SVM cost-vs-
    margin trade-off coefficient
- scoreRBFKernelSVM(svmModel, D): computes the classification scores for samples in
    the data matrix D for an SVM model svmModel (as returned by the function
    trainRBFKernelSVM) and returns an array of scores
- evaluateScores(S, L): computes an evaluation metric (e.g. minimum DCF) over the
    array of scores S with associated array of labels L

Assume that you have at your disposal a training set, already divided in model training data
(DTR, LTR)and validation data(DVAL, LVAL), and an evaluation set(DTE, LTE). DTR, DVAL
and DTE are data matrices, with samples organized as column vectors, whereas LTR, LVAL and
LTE are arrays containing the corresponding labels.

Write the Python code to train and apply an SVM classifier. In particular, the code should

- Train an SVM classifier, optimizing the value of the hyper-parameters with respect to the
    metric function evaluateScores using a single-fold cross-validation approach.
- Evaluate the selected SVM model on the evaluation data, using the provided metric.

Write an implementation of scoreRBFKernelSVM(svmModel, D). Assume that svmModel is an
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

Perfect—here’s exactly what you need, kept tight and aligned with the coding style in your course files.

# A) Single-fold CV over $(C,\gamma)$, train RBF-SVM, evaluate on test

```python
import numpy as np

# assume these are PROVIDED:
# - trainRBFKernelSVM(D, L, C, gamma) -> svmModel
# - evaluateScores(S, L) -> float  (e.g., minDCF)
# - scoreRBFKernelSVM(svmModel, D)  (implemented below)

def run_rbf_svm_single_fold_cv(DTR, LTR, DVAL, LVAL, DTE, LTE,
                               C_grid=(1e-2, 1e-1, 1, 10, 100),
                               gamma_grid=(1e-3, 1e-2, 1e-1, 1, 10)):
    """
    Trains RBF SVM models over a grid of (C, gamma) using single-fold CV
    (train on DTR/LTR, validate on DVAL/LVAL), picks the best hyperparams by
    evaluateScores, then retrains on TRAIN+VAL and evaluates on TEST.
    Returns: (best_C, best_gamma, test_scores, test_metric)
    """
    best = (None, None, np.inf)

    # --- model selection on validation ---
    for C in C_grid:
        for gamma in gamma_grid:
            model = trainRBFKernelSVM(DTR, LTR, C, gamma)
            S_val = scoreRBFKernelSVM(model, DVAL).ravel()
            metric = evaluateScores(S_val, LVAL)  # e.g., minDCF
            if metric < best[2]:
                best = (C, gamma, metric)

    best_C, best_gamma, best_val_metric = best
    print(f"[CV] best C={best_C}, gamma={best_gamma} | validation metric={best_val_metric:.4f}")

    # --- retrain on TRAIN+VAL, evaluate on TEST ---
    DTRVAL = np.hstack([DTR, DVAL])
    LTRVAL = np.hstack([LTR, LVAL])

    final_model = trainRBFKernelSVM(DTRVAL, LTRVAL, best_C, best_gamma)
    S_te = scoreRBFKernelSVM(final_model, DTE).ravel()
    test_metric = evaluateScores(S_te, LTE)
    print(f"[TEST] metric with C={best_C}, gamma={best_gamma}: {test_metric:.4f}")

    return best_C, best_gamma, S_te, test_metric
```

# B) Implementation of `scoreRBFKernelSVM(svmModel, D)`

Below matches the kernel SVM scoring used in your reference solution: build the kernel between the **support vectors** and the samples to score, then compute the weighted sum $\sum_i \alpha_i z_i k(x_i, x)$. The efficient RBF computation mirrors the broadcast trick in your material (use $\|x-y\|^2=\|x\|^2+\|y\|^2-2x^\top y$).&#x20;

```python
import numpy as np

def RBFKernel(D1: np.ndarray, D2: np.ndarray, gamma: float) -> np.ndarray:
    """
    D1, D2: (d, n1), (d, n2) with column-wise samples
    returns K with K[i,j] = exp(-gamma * ||D1[:,i] - D2[:,j]||^2)
    """
    # Fast pairwise squared distances (same pattern as in sol.py)
    n1 = D1.shape[1]
    n2 = D2.shape[1]
    n1_norm = (D1**2).sum(0)               # (n1,)
    n2_norm = (D2**2).sum(0)               # (n2,)
    # broadcasting: n1 x n2
    sq = np.add.outer(n1_norm, n2_norm) - 2.0 * (D1.T @ D2)
    return np.exp(-gamma * sq)

def scoreRBFKernelSVM(svmModel, D: np.ndarray) -> np.ndarray:
    """
    svmModel fields:
      - sv: (d, n_sv) support vectors (columns)
      - alpha: (n_sv,) Lagrange multipliers
      - labels: (n_sv,) labels in {+1, -1} for each SV
      - gamma: float, RBF hyper-parameter
    D: (d, n) samples to score (columns)
    Returns: (n,) array of decision scores (no bias term assumed).
    """
    K = RBFKernel(svmModel.sv, D, svmModel.gamma)   # (n_sv, n)
    # sum_i alpha_i * z_i * k(sv_i, x)
    w_sv = svmModel.alpha * svmModel.labels         # (n_sv,)
    S = (w_sv[:, None] * K).sum(0)                  # (n,)
    return S
```

### Notes that tie back to your files

* The RBF kernel broadcast trick and kernel-SVM scoring pattern are the same as the reference kernel SVM implementation where the final scorer returns $\sum_i \alpha_i z_i k(x_i, x)$.&#x20;
* If your metric is minDCF/actDCF, the helper functions in `bayesRisk.py` show how they’re computed from scores; your `evaluateScores` can wrap those (as the project materials do).&#x20;

If you already have a specific `evaluateScores` in your codebase, just plug it into `run_rbf_svm_single_fold_cv` as is.


### # Project - question example 6
Consider a binary classification problem, with classes labeled as 1 and 0, respectively.

Let `(DTR, LTR)`, `(DVAL, LVAL)` represent a labeled training set and a labeled validation
set. 

`DTR` and `DVAL` are 2-D numpy arrays containing the dataset samples (stored as column
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
Here’s a tight, self-contained fragment that (1) trains a **calibrated** binary classifier and (2) predicts **labels** on `DTE`.
It trains the base model on `(DTR, LTR)`, learns a calibration on **validation scores** `(DVAL, LVAL)` with prior `p`, then applies optimal Bayes thresholding to calibrated test scores:

```python
import numpy as np
import bayesRisk  # for Bayes thresholding on (calibrated) llrs

# GIVEN:
# trainClassifier(D, L) -> model
# scoreClassifier(model, D) -> 1-D scores (non-calibrated)
# trainCalibrationModel(S, L, prior) -> calModel
# applyCalibrationModel(calModel, S) -> calibrated scores (LLR-like)

def train_and_predict_calibrated(DTR, LTR, DVAL, LVAL, DTE, p=0.5):
    # 1) Train base (non-calibrated) classifier on TRAIN
    model = trainClassifier(DTR, LTR)

    # 2) Get NON-calibrated scores on VALIDATION and train calibration with target prior p
    S_val_raw = scoreClassifier(model, DVAL)              # shape (n_val,)
    calModel   = trainCalibrationModel(S_val_raw, LVAL, p)

    # 3) Score TEST, apply calibration to obtain (approx.) LLRs for the application prior p
    S_te_raw = scoreClassifier(model, DTE)                # shape (n_te,)
    S_te_cal = applyCalibrationModel(calModel, S_te_raw)  # calibrated scores (LLR-like)

    # 4) Turn calibrated scores into predicted labels via optimal Bayes rule at prior p
    #    (threshold t = -log(p/(1-p)); using the course helper for clarity)
    pred = bayesRisk.compute_optimal_Bayes_binary_llr(S_te_cal, p, Cfn=1.0, Cfp=1.0)  # returns 0/1
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)
```

Why this is correct (per course code/notes):

* After calibration, scores are treated as **LLRs**, so the **optimal Bayes decision** is obtained by comparing to the threshold $t=-\log\frac{p}{1-p}$; the helper `compute_optimal_Bayes_binary_llr` applies exactly this rule.&#x20;
* The class material also shows subtracting/adjusting for priors to obtain LLR-like scores before Bayes decisions—consistent with using calibrated scores and prior-based thresholding here.&#x20;

If you also want to report actDCF/minDCF on `DTE`, you can plug `S_te_cal` into the helpers in `bayesRisk.py`.&#x20;


### Version 2
Here you go — same logic as before, but with the Bayes decision rule coded inline (no `bayesRisk` import). The calibrated scores are interpreted as LLRs; the optimal threshold for an application with effective prior `p` is `t = -log(p/(1-p))`. We train the base classifier on `DTR,LTR`, learn the calibrator on `DVAL,LVAL` with prior `p`, and finally predict labels on `DTE`. &#x20;

```python
import numpy as np

def _bayes_threshold_from_prior(p: float) -> float:
    """
    Optimal threshold for calibrated LLR scores under effective prior p (class 1).
    t = -log(p / (1 - p))
    """
    if not (0.0 < p < 1.0):
        raise ValueError("Effective prior p must be in (0,1).")
    return -np.log(p / (1.0 - p))

def _decide_from_llr(llr_scores: np.ndarray, p: float) -> np.ndarray:
    """
    Turn calibrated LLR scores into predicted labels {0,1} using Bayes threshold.
    """
    t = _bayes_threshold_from_prior(p)
    return (llr_scores > t).astype(int)

def train_calibrated_and_predict(
    DTR: np.ndarray, LTR: np.ndarray,
    DVAL: np.ndarray, LVAL: np.ndarray,
    DTE: np.ndarray,  p: float
):
    """
    1) Train non-calibrated classifier on DTR/LTR.
    2) Train a calibration model on validation scores (DVAL/LVAL) using prior p.
    3) Score & calibrate evaluation data (DTE).
    4) Predict labels on DTE via Bayes rule with effective prior p.

    Returns
    -------
    S_te_cal : np.ndarray
        Calibrated scores (LLRs) for evaluation samples.
    y_hat    : np.ndarray
        Predicted labels {0,1} for evaluation samples.
    """
    # 1) base model
    base_model = trainClassifier(DTR, LTR)

    # 2) calibration on validation split
    S_val = scoreClassifier(base_model, DVAL)                  # non-calibrated
    cal_model = trainCalibrationModel(S_val, LVAL, prior=p)    # learns α,β (affine mapping)

    # 3) evaluation scores + calibration
    S_te = scoreClassifier(base_model, DTE)                    # non-calibrated
    S_te_cal = applyCalibrationModel(cal_model, S_te)          # calibrated LLRs

    # 4) Bayes decisions with effective prior p
    y_hat = _decide_from_llr(S_te_cal, p)

    return S_te_cal, y_hat
```

Why this is correct (in one breath):

* Score calibration (e.g., prior-weighted logistic regression) learns an affine map turning raw scores into approximate LLRs; once scores are LLR-like, the Bayes-optimal threshold for prior `p` is `-log(p/(1-p))`. &#x20;


<br>

---

# Functions code snippets
### PCA
```python
def trainPCA(D: np.ndarray, m: int) -> np.ndarray:
    """
    Train a PCA projection on data.
    Args:
        D: (d, n) data matrix with column-wise samples.
        m: target dimensionality (m <= d).
    Returns:
        P: (d, m) PCA projection matrix (columns are the top-m PCs).
    """
    # Center data
    mu = vcol(D.mean(1))
    # Compute covariance
    C = (D - mu) @ (D - mu).T / D.shape[1]  # Covariance matrix
    # SVD
    U, _ = np.linalg.svd(C)
    P = U[:, :m]  # Top-m principal components
    
    return P
```

```python
def applyPCA(P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Apply a trained PCA projection.
    Args:
        P: (d, m) PCA projection matrix.
        D: (d, n) data matrix.
    Returns:
        DP: (m, n) projected data.
    """
    return P.T @ D  # Project data onto PCA subspace
```

### LDA
```python
