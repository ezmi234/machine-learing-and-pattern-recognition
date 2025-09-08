### MVG:

- **Standard**: Lowest minDCF compared with Naive and Tied. It is indeed comparable to minDCF obtained from GMM.
  In this case, the actDCF is greater than minDCF, showing miscalibration due to the presence of non-Gaussian features.

- **Tied**: Poorer performances in terms of minDCF. The gap between minDCF and actDCF is smaller, resulting in a well
  calibrated model.

- **Naive**: Performs surprisingly well, almost matching standard MVG, which is explained by the low feature
  correlations observed in the dataset.
 
- **GMM**: We observe that the Gaussian Mixture Model (GMM) with diagonal covariance delivers the best overall performance. Specifically, when modeling class 0 with 8 components and class 1 with 32 components, it achieves a minDCF of 0.1312 and an actDCF of 0.1517. This significantly outperforms both the MVG and its variants. Suitable when features are multi-modal (e.g., features 5 and 6 form distinct clusters).


| Model         | minDCF (ideal perf.)      | actDCF (practical perf.) | Notes                                      |
|---------------|---------------------------|--------------------------|--------------------------------------------|
| **MVG**       | Low                       | ❌ Higher gap vs minDCF   | non-Gaussian features → poor calibration   |
| **Tied MVG**  | Discrete (worse than MVG) | ⭐ Close to minDCF        | Reliable (well calibrated)                 |
| **Naive MVG** | Low                       | ⭐ Close to minDCF        | Overall performs better than tied          |
| **GMM**       | ⭐ Lowest                  | ⭐ Low (close to minDCF)  | Best overall, flexible + fairly calibrated |
---
### Linear vs Non-linear SVM with regularization

- **Linear SVM**: Worst performances among the three in terms of minDCF, showing inability to capture complex patterns. The gap between minDCF and actDCF is large, resulting in a poorly calibrated model. Increasing the value of the hyperparameter C helps improve actDCF. Extremely small values of C can deteriorate the value of minDCF, whereas others show no impact.

- **Polynomial SVM**: Better performances than Linear in terms of minDCF, since it's able to capture non-linear patterns. The gap between minDCF and actDCF is the narrowest, although still showing miscalibration. In this case the impact of C is identical to Linear SVM. 
 
- **RBF SVM**: This kernel helps effectively in modeling the complex data boundaries, achieving the best minDCF (0.177). The actDCF is comparable to the model that uses Polynomial SVM. Larger values of C lead to better minDCF results across $\gamma$ values, achieving its minimum at mid-range C values. Like minDCF, also actDCF improves as the hyperparameters increase. Small values of $\gamma$ help calibration. Although, the bigger the values of C, the more irrelevant it becomes. 

| Model          | minDCF (ideal perf.) | actDCF (practical perf.)               | Regularization                                                                                  | Notes                                            |
|----------------|----------------------|----------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------|
| **Linear SVM** | Decent               | ❌ Highest gap                          | medium-High C improve actDCF                                                                    | Requires calibration/regularization              |
| **Poly SVM**   | Good                 | ❌ High gap vs minDCF                   | same as linear                                                                                  | Best gap but still miscalibrated                 |
| **RBF SVM**    | ⭐ Best among SVMs    | ❌ High gap vs minDCF (similar to Poly) | Two Hyperparameters $\gamma$, C; minimum values of minDCF across $\gamma$ values at mid-range C | Slightly worse, needs calibration/regularization |
---
### Logistic regression with regularization

- **Standard LR**: Good, but sensitive to the choice of regularization parameter $\lambda$. Small $\lambda$ → risk of overfitting. Large $\lambda$ → underfitting (poor separation). The gap between minDCF and actDCF is visible, showing miscalibration. Increasing $\lambda$ stabilizes the actDCF but hurts the discriminative power.

- **Weighted LR**: Weighting by the effective prior (π̃ = 0.1), the model performs similarly to standard LR — the gap does not change significantly. Embedding prior knowledge during training in this context suggests that the regularization might not be very effective.
 
- **Quadratic LR**: It achieves the lowest minDCF among LR variants, because quadratic feature expansion captures non-linear relationships. Still shows miscalibration (gap between minDCF and actDCF remains), though performance in terms of minDCF is the best. Combining with preprocessing (centering, PCA) does not improve minDCF or the calibration.

| Model            | minDCF (ideal perf.) | actDCF (practical perf.)               | Regularization                                                                                  | Notes                                            |
|------------------|----------------------|----------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------|
| **Standard LR**  | Decent               | ❌ High gap vs minDCF                   | medium-High C improve actDCF                                                                    | Requires calibration/regularization              |
| **Weighted LR**  | Similar to Standard  | ❌ High gap vs minDCF                   | same as linear                                                                                  | Best gap but still miscalibrated                 |
| **Quadratic LR** | ⭐ Best among SVMs    | ❌ High gap vs minDCF (similar to Poly) | Two Hyperparameters $\gamma$, C; minimum values of minDCF across $\gamma$ values at mid-range C | Slightly worse, needs calibration/regularization |

---
<br>

### Summary Table
| Model        | parameters                                                          | effective prior | minDCF | actDCF | Regularization | Notes                                                                                                                                |
|--------------|---------------------------------------------------------------------|-----------------|--------|--------|----------------|--------------------------------------------------------------------------------------------------------------------------------------|
| MVG          |                                                                     | 0.1             | 0.253  | 0.305  | N/A            | Low minDCF, still miscalibrated due to non-Gaussian features(needed MultiModal model)                                                |
| Tied MVG     |                                                                     | 0.1             | 0.38   | 0.396  | N/A            | Poorer performances in terms of minDCF. The gap between minDCF and actDCF is smaller, resulting in a well calibrated model           |
| Naive MVG    |                                                                     | 0.1             | 0.263  | 0.302  | N/A            | performs surprisingly well, almost matching standard MVG, which is explained by the low feature correlations observed in the dataset |
| Standard LR  |                                                                     | 0.1             | 0.364  | 0.402  | λ              | small λ(weak regularization) small actDCF, big λ big actDCF                                                                          |
| Weighted LR  |                                                                     | 0.1             | \\\\\  | \\\\\  | λ              | same as standard LR                                                                                                                  |
| Quadratic LR |                                                                     | 0.1             | 0.247  | 0.277  | λ              | quadratic better performance(minDCF), regularization does not affect performances(minDCF) but affects calibration(actDCF)            |
| Linear SVM   | N/A                                                                 | 0.1             | 0.358  | 0.489  | C              |                                                                                                                                      |
| Poly SVM     | d=2                                                                 | 0.1             | 0.221  | 0.389  | C              | Good performance but still not calibrated(Better calibration big values of C(close to 1), worst small value of C)                    |
| RBF SVM      | N/A                                                                 | 0.1             | 0.177  | 0.428  | C, γ           | RBF delivers good performance but not calibrated                                                                                     |
| GMM          | components {class 0: 8; class 1: 32}, $\psi = 0.01$, $\alpha = 0.1$ | 0.1             | 0.131  | 0.152  | N/A            | best performance and calibration, in particular diagonal covariance respect to full covariance                                       |
