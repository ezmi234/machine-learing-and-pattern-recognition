# Exam 26/06/2025

### Theory - question 1 (12 pt)

Describe in detail the multivariate Gaussian classifier, covering the following aspects:
- Model assumptions
- Estimation of the model parameters
- How the model can be employed to perform inference (i.e. to classify a test sample) for both multi-class and binary problems
- The form of decision rules for binary problems
- Naive Bayes and Tied Covariance variants of the model focusing on:
	- Differences with the standard (unconstrained) model in terms of assumption and decision rules
	- Benefit and limitation with respect to the unconstrained model

### Theory - question 2 (12 pt)
Describe the binary logistic regression model for classification, covering the following aspects:
- Classification rule of the model
- Probabilistic interpretation of the model and of its classification score
- Estimation of the model parameters and possible interpretations of the training objective function
- How the model can extended to perform non-linear classification
- How the model can extended and applied to address score calibration issues

---
### Project - question 1 (5 pt)

Briefly explain the relative performance on the project validation set of different SVM kernels (including linear models), in light of the characteristics of the kernel and of the characteristics of the dataset. Analyse the impact of the regularisation coefficient on the results of the SVM classifiers on the same validation set.

### Project - question 2 (3 pt)

Given the following functions (assume these functions are already implemented):
- `trainPCA`: trains a PCA model
- `applyPCA`: applies a PCA model to some data
- `trainLDA`: trains a LDA model
- `applyLDA`: applies a LDA model to some data
- `evaluateScores(S, L)`: computes a performance metric (e.g. minimum DCF for a predefined application) over the score array S with label L
1. Provide possible signatures (prototype) for the `trainPCA`, `applyPCA`, `trainLDA`, `applyLDA` and `evaluateScores(S, L)` functions, briefly explaining the function parameters and the return value.
2. Using these functions, write a short Python program to train and apply an LDA binary classifier. The classifier should simply output a classification <u>score</u> (i.e. you are <u>not</u> required to provide predicted labels or select a suitable score threshold). The performance of the classifier can be evaluated through the provided function `evaluateScores`. Assume that you have at your disposal a training set, already directed in training data (DTR, LTR) and validation data (DVAL, LVAL) and an evaluation set (DTE, LTE). DTR, DVAL and DTE are data matrices, with samples organised as columns vectors, whereas LTR, LVAL and LTE are arrays containing the corresponding labels. The program should employ in an appropriate way the provided data to train the classification model and to select an optimal value for the PCA dimensionality scores and the corresponding performance for the evaluation set.
