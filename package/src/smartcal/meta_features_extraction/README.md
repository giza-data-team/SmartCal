# MetaFeaturesExtractor

**MetaFeaturesExtractor** is a Python module designed to extract meta-features for a given set of true labels and predicted probabilities from the calibration set. It computes a variety of characteristics, including statistical summaries, class imbalance measures, classification metrics, calibration scores, and pairwise distribution distances. this module is used for getting the meta-dataset from the results of knowledge base experiments, which is then used to train a meta-model that predicts the most suitable calibration method for a given set of true labels and predicted probabilities from the calibration set.

---

## Extracted Meta-Features 

| **Category**            | **Feature Name in CSV**                     | **Description**                                                                 |
|-------------------------|---------------------------------------------|---------------------------------------------------------------------------------|
| **Dataset Info**        | `Dataset_name`                              | Name of the dataset **(for meta-data analysis only)**                                                           |
|                         | `Model_name`                                | Name of the classification model **(for meta-data analysis only)**                                                  |
|                         | `Best_Cal`                                  | Name of the best calibration method according to the minimum ece                                            |
|                         | `num_classes`                               | Number of classes in the dataset                                               |
|                         | `num_instances`                             | Number of data instances in the calibration set                                                       |
|                         | `class_imbalance_ratio`                     | Ratio of most common class to least common class                               |
| **Prediction Distribution** | `actual_predictions_entropy`            | Entropy of predicted class label distribution                                  |
| **Confidence Statistics** | `Confidence_Mean`                         | Mean confidence value (1 - true class prob)                                    |
|                         | `Confidence_Median`                         | Median of confidence values                                                    |
|                         | `Confidence_Std`                            | Standard deviation of confidence values                                        |
|                         | `Confidence_Var`                            | Variance of confidence values                                                  |
|                         | `Confidence_Entropy`                        | Entropy of confidence values                                                   |
|                         | `Confidence_Skewness`                       | Skewness of confidence values                                                  |
|                         | `Confidence_Kurtosis`                       | Kurtosis of confidence values                                                  |
|                         | `Confidence_Min`                            | Minimum confidence                                                             |
|                         | `Confidence_Max`                            | Maximum confidence                                                             |
| **Classification Metrics** | `Classification_Accuracy`               | Accuracy of predictions                                                        |
|                         | `Classification_Log_loss`                   | Log loss of predicted probabilities                                            |
|                         | `Classification_Precision_Micro`            | Micro-average precision                                                        |
|                         | `Classification_Precision_Macro`            | Macro-average precision                                                        |
|                         | `Classification_Precision_Weighted`         | Weighted-average precision                                                     |
|                         | `Classification_Recall_Micro`               | Micro-average recall                                                           |
|                         | `Classification_Recall_Macro`               | Macro-average recall                                                           |
|                         | `Classification_Recall_Weighted`            | Weighted-average recall                                                        |
|                         | `Classification_F1_Micro`                   | Micro-average F1 score                                                         |
|                         | `Classification_F1_Macro`                   | Macro-average F1 score                                                         |
|                         | `Classification_F1_Weighted`                | Weighted-average F1 score                                                      |
| **Calibration Metrics** | `ECE_before`                                | Expected Calibration Error (before calibration)                                |
|                         | `MCE_before`                                | Maximum Calibration Error (before calibration)                                 |
|                         | `ConfECE_before`                            | Confidence-based ECE                                                           |
|                         | `brier_score_before`                        | Brier score (mean squared error of probabilistic forecasts)                    |
| **Distance Metrics (Pairwise between class distributions)** | `Wasserstein_Mean`              | Mean Wasserstein distance between class-wise predicted probability distributions |
|                                                             | `Wasserstein_Std`               | Standard deviation of Wasserstein distances                                      |
|                                                             | `Wasserstein_Entropy`           | Entropy of Wasserstein distances                                                 |
|                                                             | `Wasserstein_Skewness`          | Skewness of Wasserstein distances                                                |
|                                                             | `Wasserstein_Kurtosis`          | Kurtosis of Wasserstein distances                                                |
|                                                             | `Wasserstein_Min`               | Minimum Wasserstein distance                                                     |
|                                                             | `Wasserstein_Max`               | Maximum Wasserstein distance                                                     |
|                                                             | `KL_Divergence_Mean`            | Mean KL divergence                                                               |
|                                                             | `KL_Divergence_Median`          | Median KL divergence                                                             |
|                                                             | `KL_Divergence_Std`             | Standard deviation of KL divergences                                             |
|                                                             | `KL_Divergence_Var`             | Variance of KL divergences                                                       |
|                                                             | `KL_Divergence_Entropy`         | Entropy of KL divergences                                                        |
|                                                             | `KL_Divergence_Skewness`        | Skewness of KL divergences                                                       |
|                                                             | `KL_Divergence_Kurtosis`        | Kurtosis of KL divergences                                                       |
|                                                             | `KL_Divergence_Min`             | Minimum KL divergence                                                            |
|                                                             | `KL_Divergence_Max`             | Maximum KL divergence                                                            |
|                                                             | `Jensen_Shannon_Mean`           | Mean Jensen-Shannon distance                                                     |
|                                                             | `Jensen_Shannon_Median`         | Median Jensen-Shannon distance                                                   |
|                                                             | `Jensen_Shannon_Std`            | Standard deviation of Jensen-Shannon distances                                   |
|                                                             | `Jensen_Shannon_Var`            | Variance of Jensen-Shannon distances                                             |
|                                                             | `Jensen_Shannon_Entropy`        | Entropy of Jensen-Shannon distances                                              |
|                                                             | `Jensen_Shannon_Skewness`       | Skewness of Jensen-Shannon distances                                             |
|                                                             | `Jensen_Shannon_Kurtosis`       | Kurtosis of Jensen-Shannon distances                                             |
|                                                             | `Jensen_Shannon_Min`            | Minimum Jensen-Shannon distance                                                  |
|                                                             | `Jensen_Shannon_Max`            | Maximum Jensen-Shannon distance                                                  |
|                                                             | `Bhattacharyya_Mean`            | Mean Bhattacharyya distance                                                      |
|                                                             | `Bhattacharyya_Median`          | Median Bhattacharyya distance                                                    |
|                                                             | `Bhattacharyya_Std`             | Standard deviation of Bhattacharyya distances                                    |
|                                                             | `Bhattacharyya_Var`             | Variance of Bhattacharyya distances                                              |
|                                                             | `Bhattacharyya_Entropy`         | Entropy of Bhattacharyya distances                                               |
|                                                             | `Bhattacharyya_Skewness`        | Skewness of Bhattacharyya distances                                              |
|                                                             | `Bhattacharyya_Kurtosis`        | Kurtosis of Bhattacharyya distances                                              |
|                                                             | `Bhattacharyya_Min`             | Minimum Bhattacharyya distance                                                   |
|                                                             | `Bhattacharyya_Max`             | Maximum Bhattacharyya distance                                                   |
