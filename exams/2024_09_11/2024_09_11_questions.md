# Exam 11/09/2024

### Theory - question 1 

Considering the Linear Discriminant Analysis (LDA) approach for <u>binary</u> classification and the Tied MVG  <u>binary</u> classifier, detail:

- Model formulation, training objective and inference procedure (i.e. how to employ the model for classification) of the LDA classifier
- Model assumptions, training objective and inference procedure of the Tied MVG classifier
- The relationship between the two models
- The form of the decision rules of LDA and Tied MVG binary classifiers

For <u>multiclass problems</u>, LDA can be employed as a <u>dimensionality reduction</u> technique. In this context, briefly explain the objective function of LDA and the limitations of the approach.

### Theory - question 2 

Describe the Support Vector Machine classifier, covering the following aspects:

- Classification rule of SVM and interpretation of the SVM score
- The concept of the margin
- Primal (both constrained convex quadratic programming and hinge loss) and dual formulation of the objective function, and the relationship between the primal and the dual solutions
- SVMs for non linear classification

Both logistic regression and Support Vector Machine (SVM) ca be interpreted as risk minimization approaches.

- Compare the objective functions of the two models

---

### Project - question 1

Briefly compare, in the light of the characteristics of the clasVsifiers and the characteristics of the training and the validation datasets, the relative performance of MVG, Tied MVG and GMM models on the project validation data.

### Project - question 2 

Consider a binary classification problem, with classes labeled as 1 and 0, respectively.

Let `(DTR, LTR)`, `(DVAL, LVAL)` represent a labeled training set and a labeled validation
set. `DTR` and `DVAL` are 2-D numpy arrays containing the dataset samples (stored as column
vectors), whereas `LTR` and `LVAL` are 1-D numpy arrays containing the sample labels. Let also
`DTE` represent the dataset matrix (again, a 2-D numpy array) containing the samples that our
application should classify.

Write a Python code fragment that:

1. trains a <u> calibrated</u> binary classifier
2. performs inference (i.e. <u> computes predicted labels</u>) on the evaluation data

You can assume that the following functions have been defined:

- `trainClassifier(D, L)`: train a non-calibrated classification model (e.g., an SVM or
    an LDA classifier) on the training matrix `D` with associated labels array `L`, and return
    a python object containing the trained model (assume that the model does not contain
    tunable hyper-parameters)
- `scoreClassifier(model, D)`: compute the <u>non-calibrated</u> classification scores for model
    model (as returned by `trainClassifier`) for the samples in data matrix `D` and return
    a 1-D array of scores
- `trainCalibrationModel(S, L, prior)`: train a calibration model on the 1-D array of
    scores S, with associated array of labels `L`, for a binary application with prior `prior` for
    class 1, and return a python object containing the trained model
- `applyCalibrationModel(calModel, S)`: apply the calibration model `calModel` (as re-
    turned by `trainCalibrationModel`) to the 1-D array of scores S, and return a 1-D array
    of calibrated scores

NOTE: assume that the target application is characterized by an effective prior `p` for class 1.

You are <u> not required</u> to tune the calibration model hyper-parameter `prior`, but you can assume
that the calibration model can be trained using the target application prior `p`.

