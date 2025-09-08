# SVM - Extensions
1. **Logistic Loss vs Hinge Loss**
2. **In soft margin how C influences the classification...** 
3. **Which points influence the model (SVs) how this differ from LR**
4. **How to choose the Kernel** 
5. **Probabilistic interpretation, how to calibrate the model** 
6. **Multiclass SVM** 
7. **Why is logistic regression often used as a score calibration tool for other classifiers?**
8. **Why is SVM not used for score calibration?**
9. **Why is SVM not used for probability estimation?**
10. **Why is SVM not used for density estimation?**
11. **Compare logistic regression with Linear Discriminant Analysis (LDA). Under what assumptions do they coincide?**
12. **What happens to logistic regression when data is linearly separable? Why do we need regularization?**


### 🔹 Core mechanics

1. *“Starting from the definition of the margin, derive why the SVM optimization problem minimizes $\tfrac{1}{2}\|w\|^2$.”*
   → Tests geometric intuition → algebraic link between margin maximization and regularization.

2. *“Explain why only support vectors affect the SVM decision boundary. What happens if a point is far away from the margin?”*
   → Tests understanding of hinge loss flat region and sparsity of the solution.

---

### 🔹 Optimization view

3. *“Write the primal and dual optimization problems for a soft-margin SVM. Discuss the role of slack variables and the regularization parameter $C$.”*
   → Tests ability to recall and explain primal/dual.

4. *“What conditions connect the primal and dual solutions (KKT conditions)? Why are they important in SVMs?”*
   → Tests understanding of convex optimization.

---

### 🔹 Kernels and non-linear classification

5. *“What is the kernel trick in SVMs, and why is it useful for non-linear classification?”*
   → Tests understanding of kernels and implicit feature maps.

6. *“Compare polynomial and Gaussian RBF kernels in terms of flexibility and risk of overfitting.”*
   → Tests knowledge of kernel properties and practical implications.

---

### 🔹 Comparisons

7. *“Compare hinge loss (SVM) and logistic loss (logistic regression). How do they behave for correctly and incorrectly classified samples?”*
   → Common “compare and contrast” question, showing risk minimization perspective.

8. *“Both logistic regression and SVMs can be seen as large-margin classifiers. In what sense is logistic regression a ‘soft’ margin method compared to SVMs?”*
   → Tests nuanced understanding of the link between the two.

---

### 🔹 Generalization and practice

9. *“How does the parameter $C$ affect the decision boundary in soft-margin SVMs? What happens for very large or very small $C$?”*
   → Practical question on bias–variance trade-off.

10. *“SVMs are not probabilistic models. What does this mean, and how can their scores be calibrated to probabilities?”*
    → Bridges SVMs to calibration and evaluation (connects with actDCF/minDCF).

---

### 🔹 Trickier variants

11. *“Why is the SVM optimization problem convex, and what guarantees does convexity provide for training?”*
    → Optimization theory.

12. *“Explain why SVMs are particularly suitable for high-dimensional data (e.g., text classification).”*
    → Application-based reasoning: margin maximization, dependence only on support vectors, kernel trick.

13. *“Describe how SVMs can be extended to multi-class problems. Mention at least two strategies.”*
    → Tests knowledge of one-vs-one vs one-vs-all.
