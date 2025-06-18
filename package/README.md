# smartcal – Auto-Calibration for Machine Learning
[![PyPI version](https://img.shields.io/pypi/v/smartcal.svg)](https://pypi.org/project/smartcal/)
![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)

**smartcal** is a modular and extensible Python package that provides automated, metric-driven calibration of classification models. It leverages meta-learning to recommend and tune the best calibrator from a rich suite of calibration algorithms. smartcal supports evaluation with popular calibration metrics.

---

## Features

### 1. SmartCal Engine
- **Meta-learning-powered recommendation** of calibration methods based on dataset characteristics.
- Automated **hyperparameter tuning via Bayesian Optimization**.
- Provides a unified interface for fitting and applying the best calibrator using:
    ```python
    from smartcal import SmartCal

    # Initialize SmartCal with your preferred metric
    smartcal = SmartCal(metric='ECE')  # Supports metrics: ECE, MCE, ConfECE, brier_score, log_loss

    # Step 1: Get top 3 recommended calibration methods
    recommended = smartcal.recommend_calibrators(y_true, predictions_prob, n=3) 

    # Step 2: Fit and retrieve the best calibrator
    best_calibrator = smartcal.best_fitted_calibrator(y_true, predictions_prob, n_iter=20)

    # Step 3: Use the calibrator
    calibrated_probs = best_calibrator.predict(predictions_prob)
    ```
- Note: The maximum number of supported calibrators is 12, not 13. This is because we do not include the Probability Tree Calibrator in the default meta-calibration pipeline, as it operates differently by incorporating data features in addition to model outputs. However, you can still use it independently via the calibration_algorithms module.

### 2. Calibration Algorithms
- Supports **diverse calibration methods**, including:
  - Parametric: Temperature Scaling, Platt, Vector, Matrix, Beta, Dirichlet, Adaptive TS
  - Non-parametric: Isotonic, Histogram, Empirical Binning
  - Hybrid: Meta Calibration, Mix-and-Match Calibration
- Each calibrator implements standard `.fit()` and `.predict() and metadata` APIs.

### 3. Calibration Metrics
- Built-in evaluation metrics to assess calibration quality:
  - **ECE** – Expected Calibration Error
  - **MCE** – Maximum Calibration Error
  - **ConfECE** – Confidence-Binned ECE
  - **Brier Score** – Proper scoring rule measuring accuracy of probabilistic predictions
  - **Calibration Curves** – for visual inspection of reliability

---

## Package Structure

```smartcal/
├── calibration_algorithms/       # All calibration method implementations
├── metrics/                      # Calibration evaluation metrics
├── config/                       # Configuration enums and constants
├── meta_model/                   # Meta-learning recommendation engine
├── meta_features_extraction/     # Meta-feature computation utilities
├── bayesian_optimization/        # Bayesian optimization computation utilities
├── utils/                        # Helper functions and validation
├── smartcal/                     # Core SmartCal meta-calibration engine
└── __init__.py
```

---

## Supported Calibration Algorithms

1. Empirical Binning  
2. Isotonic Regression  
3. Temperature Scaling  
4. Beta Calibration  
5. Dirichlet Calibration  
6. Platt Scaling  
7. Vector Scaling  
8. Matrix Scaling  
9. Adaptive Temperature Scaling  
10. Histogram Calibration  
11. Mix-and-Match Calibration  
12. Meta Calibration  
13. Probability tree Calibration

Each calibrator supports `.fit()` and `.predict()` with `(n_samples, n_classes)` formatted input.

---

## Calibration Metrics

smartcal provides implementations for:

- **ECE (Expected Calibration Error)**
- **MCE (Maximum Calibration Error)**
- **Confidence-ECE** (for threshold-based confidence bins)
- **Brier Score** (proper scoring rule measuring accuracy of probabilistic predictions)
- **Calibration Curve Plotting** (for visualization)

---

## Documentation & Usage

For full documentation and usage guidance, please refer to this colab notebook: [smartcal.ipynb](https://colab.research.google.com/drive/19Tj2z7GfgvQb5Dwjiryg0C0DoieXen2j?usp=sharing)

---

## Collaborators

This project is a collaborative effort developed with support from **Giza Systems**. We would like to acknowledge the following contributors:

### Core Team
- **Mohamed Maher** - [GitHub](https://github.com/mmaher22) | [PyPI](https://pypi.org/user/mmaher22/) | [Email](mailto:m.maher525@gmail.com)
- **Mariam Elseedawy** - [GitHub](https://github.com/Seedawy200) | [PyPI](https://pypi.org/user/Seedawy200/) | [Email](mailto:mariam.elseedawy@gmail.com)
- **Osama Fayez** - [GitHub](https://github.com/osamaoun97) | [PyPI](https://pypi.org/user/osamaoun97/) | [Email](mailto:osamaoun997@gmail.com)
- **Youssef Medhat** - [GitHub](https://github.com/yossfmedhat) | [PyPI](https://pypi.org/user/yossfmedhat/) | [Email](mailto:yossfmedhat@gmail.com)
- **Yara Marei** - [GitHub](https://github.com/yaramostafa) | [PyPI](https://pypi.org/user/yaramostafa/) | [Email](mailto:yaramostaffa500@gmail.com)
- **Abdullah Ibrahim** - [GitHub](https://github.com/BidoS) | [PyPI](https://pypi.org/user/abdullah.ibrahim/) | [Email](mailto:abdullahibrahim544@gmail.com)
- **Radwa ElShawi** - [Email](mailto:radwa.elshawi@ut.ee)

---

## Citation

If you use `SmartCal` in your research or publication, please cite it as:


**BibTeX:**
```bibtex
@inproceedings{
abdelrahman2025smartcal,
title={SmartCal: A Novel Automated Approach to Classifier Probability Calibration},
author={Mohamed Maher Abdelrahman and Mariam Magdy Elseedawy and Osama Fayez Oun and Youssef Medhat and Yara Mostafa Marei and Abdullah Ibrahim and Radwa Mohamed El Shawi},
booktitle={AutoML 2025 Methods Track},
year={2025},
url={https://openreview.net/forum?id=XPtubBurd2}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) file for details.
