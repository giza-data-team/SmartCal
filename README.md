# SmartCal

## Project Overview

**smartcal** is an open-source AutoML framework designed to automate the selection and tuning of post-hoc calibration methods for classification models. It addresses the critical challenge of obtaining accurate probability estimates—essential for high-stakes domains like medical diagnosis, fraud detection, and autonomous systems—by dynamically recommending the best calibration strategy from a pool of **12 methods** based on dataset characteristics and classifier behavior. The framework provides a comprehensive suite of 13 calibration methods and evaluation metrics including Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Confidence ECE, Brier Score, and calibration curves.

### Goals

- **Automate Calibration**: Streamline the calibration process of classification models to improve reliability with minimal user intervention.
- **Support most of Calibration Methods**: Unified interface for 13 calibration algorithms including Temperature Scaling, Platt Scaling, Beta Calibration, Isotonic Regression, Vector/Matrix Scaling, Dirichlet, Meta-Calibration, Histogram Binning, Adaptive Temperature Scaling, Mix & Match, and Probability Trees.
- **Provide Calibration Metrics**: Evaluate calibration methods using metrics like Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Confidence ECE, and Execution Time Overhead.
- **Meta-Learning**: Develop a meta-model that recommends the best calibration method based on dataset characteristics (e.g., kurtosis, skewness, histogram binning).
  
## Methodology

The project follows a systematic approach to build and evaluate calibration algorithms:

1. **Dataset Collection**: Gather datasets across binary and multi-class classification tasks, ensuring a diverse range of classification problems.
2. **Knowledge Base Construction**: Store detailed metadata for each experiment to facilitate later use in the meta-model.
3. **Meta-Models Development**: Train a meta-learning model that suggests the most appropriate calibration method based on dataset characteristics.
4. **Performance Benchmarking**: Evaluate and compare different calibration techniques using defined metrics, and benchmark the meta-model against other search strategies (e.g., random search) for selecting the best method.

---
## Repository Structure: SmartCal
```python
SmartCal/
├── package/                   # Main package directory
│   ├── src/smartcal/          # Core smartcal package source code
│   │   ├── bayesian_optimization/ # Bayesian optimization implementation 
│   │   ├── calibration_algorithms/ # Various calibration method implementations
│   │   ├── config/            # Configuration files and settings
│   │   ├── meta_features_extraction/ # Feature extraction utilities
│   │   ├── meta_model/        # Meta-learning model implementation
│   │   ├── metrics/           # Calibration evaluation metrics
│   │   ├── smartcal/          # Core smartcal implementation
│   │   ├── utils/             # Utility functions and helpers
│   │   └── __init__.py        # Package initialization file
│   ├── examples/              # Usage examples and demonstrations
│   ├── dist/                  # Distribution files
│   ├── pyproject.toml         # Package configuration and dependencies
│   ├── LICENSE                # Package license
│   └── README.md              # Package documentation
├── classifiers/               # Model implementations for different data types
├── data_preparation/          # Data preprocessing and preparation
├── datasets/                  # Dataset storage and management
├── experiment_manager/        # Experiment tracking and management on databases
├── experiments/               # Experiment configurations and scripts
├── meta_data_extraction/      # Script for data meta features extraction from knowledgebase
├── meta_models_training/      # Training meta model (note: renamed from meta_model_training)
├── pipeline/                  # Processing pipelines for different data types
├── run_experiments/           # Scripts for running experiments
├── tests/                     # Test suite
├── docker-compose.yml         # Docker configuration for running PostgreSQL database and pgAdmin
├── requirements.txt           # Python packages and dependencies required to run the project
├── .gitignore                 # Git ignore file
└── __init__.py                # Root package initialization
```

### Key Components Description

#### `package/`
- Main package directory containing the smartcal distribution
- Contains source code, examples, configuration, and package metadata
- Houses the complete smartcal implementation with all supporting modules

#### `package/src/smartcal/`
- Core package containing the main implementation
- Houses the SmartCal algorithm and supporting modules 
- Contains all calibration algorithms, meta-features extractor, meta-model and metrics

#### `classifiers/`
- Various classifier implementations
- Support for different data types (image, language, tabular)

#### `data_preparation/`
- Data info for different types of data
- Data splitting and preprocessing pipelines

#### `datasets/`
- Dataset storage and management
- Data loading utilities

#### `experiment_manager/`
- Experiment tracking functionality
- Results saving and management on databases

#### `experiments/`
- Experiment configurations
- Test scripts and evaluation protocols

#### `meta_data_extraction/`
- Script to extract meta features from knowledge base

#### `meta_models_training/`
- Script to train One vs. All meta model (renamed from meta_model_training)

#### `pipeline/`
- End-to-end pipelines for automated calibration and hyperparameter optimization

#### `run_experiments/`
- Scripts for running experiments
- Automated experiment execution utilities

#### `tests/`
- Unit tests for all components

---

## Installation and Setup

### Prerequisites
- Python = 3.12.7
- Docker
- PostgreSQL

### Environment Setup

#### Step 1: Clone the Repository

```bash
git clone https://github.com/giza-data-team/SmartCal.git
cd SmartCal
```

#### Step 2: Create and Activate Virtual Environment

**Option A: Using Conda (Recommended)**
```bash
# Create environment
conda create -n autocal python=3.12

# Activate environment
conda activate autocal
```

**Option B: Using venv**

- **Windows (PowerShell/Command Prompt):**
  ```bash
  # Create environment
  python -m venv autocal
  
  # Activate environment (PowerShell)
  .\autocal\Scripts\Activate.ps1
  
  # OR Activate environment (Command Prompt)
  .\autocal\Scripts\activate.bat
  ```

- **macOS/Linux:**
  ```bash
  # Create environment
  python3.12 -m venv autocal
  
  # Activate environment
  source autocal/bin/activate
  ```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**PyTorch Installation (choose based on your system):**

• **Linux or Windows + NVIDIA GPU:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

• **CPU-only (all platforms, including macOS):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

**Alternative: Install PyTorch via conda**
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 -c pytorch
```

#### Step 4: Set Python Path

- **Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH="package/src"
  ```

- **Windows (Command Prompt):**
  ```cmd
  set PYTHONPATH=package/src
  ```

- **macOS/Linux:**
  ```bash
  export PYTHONPATH=package/src
  ```

**Note:** For persistent environment variables, consider adding the PYTHONPATH to your shell profile (`.bashrc`, `.zshrc`, etc.) or using conda environment variables.

### Quick Start

To verify your installation, run the example scripts:

#### Basic Examples
```bash
# Core SmartCal functionality
python package/examples/smartcal_example.py

# Metrics evaluation
python package/examples/metrics_example.py
```

#### Calibration Algorithm Examples
```bash
# Temperature-based calibration
python package/examples/calibration_algorithms_examples/temperature_scaling_example.py
python package/examples/calibration_algorithms_examples/adaptive_temperature_scaling_example.py

# Scaling methods
python package/examples/calibration_algorithms_examples/vector_scaling_example.py
python package/examples/calibration_algorithms_examples/matrix_scaling_example.py

# Traditional methods
python package/examples/calibration_algorithms_examples/platt_example.py
python package/examples/calibration_algorithms_examples/isotonic_example.py

# Advanced methods
python package/examples/calibration_algorithms_examples/beta_example.py
python package/examples/calibration_algorithms_examples/dirichlet_example.py
python package/examples/calibration_algorithms_examples/histogram_example.py

# Meta and ensemble methods
python package/examples/calibration_algorithms_examples/meta_example.py
python package/examples/calibration_algorithms_examples/mix_and_match_example.py

# Tree-based methods
python package/examples/calibration_algorithms_examples/probability_tree_example.py
python package/examples/calibration_algorithms_examples/empirical_binning_example.py
```

### Database Setup

SmartCal uses PostgreSQL to store experiment results and meta-learning data. Follow these steps to set up the database:

#### Step 1: Start PostgreSQL Container

```bash
# Start PostgreSQL using Docker
docker-compose up -d
```

#### Step 2: Configure Database Credentials

Database configurations such as port number, host, username, and password are located in:
```
config/resources/conf/config-dev.env
```

Make sure these settings match your Alembic and application settings:
- `SSH_ENABLED=false`
- `DB_PASSWORD=password`

#### Step 3: Initialize Database Schema

**Create migrations directory:**

- **Windows (PowerShell/Command Prompt):**
  ```bash
  mkdir experiment_manager\migrations\versions
  ```

- **macOS/Linux:**
  ```bash
  mkdir -p experiment_manager/migrations/versions
  ```

**Generate and apply migrations:**
```bash
# Generate initial migration
alembic -c experiment_manager/alembic.ini revision --autogenerate -m "Initial migration"

# Apply migrations
alembic -c experiment_manager/alembic.ini upgrade head
```

This will apply all migrations and create your database tables.

#### Step 4: Verify Setup (Optional)

To verify the database setup:

```bash
# Connect to PostgreSQL
psql -h localhost -U admin -d Calibration_db -p 5432

# Common psql commands:
# \dt - List tables
# \d+ table_name - Describe table
# \q - Exit psql
```

**Note:** For Windows users without native psql, you can use pgAdmin (included in docker-compose) or install PostgreSQL client tools.

## Documentation for Each Component

This section provides comprehensive documentation for all SmartCal components. Each component includes detailed READMEs, usage examples, and API references.

### Core Framework Components

| Component | Description | Status | Documentation |
|-----------|-------------|--------|---------------|
| **SmartCal Engine** | Main calibration framework with automated method selection | Ready | [`package/src/smartcal/smartcal/`](package/src/smartcal/smartcal/README.md) |
| **Calibration Algorithms** | Implementation of 12 post-hoc calibration methods | Ready | [`package/src/smartcal/calibration_algorithms/`](package/src/smartcal/calibration_algorithms/README.md) |
| **Meta-Features Extraction** | Feature extraction for meta-learning | Ready | [`package/src/smartcal/meta_features_extraction/`](package/src/smartcal/meta_features_extraction/README.md) |
| **Meta-Model** | Automated calibration method recommendation | Ready | [`package/src/smartcal/meta_model/`](package/src/smartcal/meta_model/README.md) |
| **Bayesian Optimization** | Hyperparameter tuning for calibrators | Ready | [`package/src/smartcal/bayesian_optimization/`](package/src/smartcal/bayesian_optimization/README.md) |
| **Evaluation Metrics** | Calibration performance metrics (ECE, MCE, Brier Score) | Ready | [`package/src/smartcal/metrics/`](package/src/smartcal/metrics/README.md) |

### Package & Distribution

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **Main Package** | Complete SmartCal package with examples and configuration | [`package/`](package/README.md) |
| **Examples** | Usage examples and demonstrations for all components | [`package/examples/`](package/examples/README.md) |
| **Configuration** | Settings and configuration management | [`package/src/smartcal/config/`](package/src/smartcal/config/README.md) |

### Data & Experiments

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **Dataset Management** | Dataset download, storage, and organization | [`datasets/`](datasets/Dataset_Download_Instructions.md) |
| **Data Preprocessing** | Data preparation pipelines for different domains | [`data_preparation/preprocessors/`](data_preparation/preprocessors/README.md) |
| **Meta-Data Extraction** | Extract meta-features from knowledge base | [`meta_data_extraction/`](meta_data_extraction/README.md) |
| **Meta-Model Training** | Train recommendation models on collected data | [`meta_models_training/`](meta_models_training/README.md) |

### Evaluation & Benchmarking

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **End-to-End Evaluation** | Complete pipeline evaluation and benchmarking | [`experiments/end_to_end_eval/`](experiments/end_to_end_eval/README.md) |
| **Meta-Model Evaluation** | Meta-model performance assessment | [`experiments/meta_model_eval/`](experiments/meta_model_eval/README.md) |
| **Experiment Execution** | Scripts for running various experiments | [`run_experiments/`](run_experiments/README.md) |

### Infrastructure & Tools

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **Experiment Manager** | Database integration and experiment tracking | [`experiment_manager/`](experiment_manager/README.md) |
| **Processing Pipelines** | Automated pipelines for different data types | [`pipeline/`](pipeline/README.md) |
| **Classifiers** | Pre-implemented classifiers for different domains | [`classifiers/`](classifiers/README.md) |
| **Testing Suite** | Comprehensive tests for all components | [`tests/`](tests/README.md) |

### Quick Navigation

**For New Users:**
1. Start with [`package/examples/`](package/examples/README.md) for basic usage
2. Review [`package/src/smartcal/smartcal/`](package/src/smartcal/smartcal/README.md) for core functionality
3. Explore [`package/src/smartcal/calibration_algorithms/`](package/src/smartcal/calibration_algorithms/README.md) for available methods

**For Researchers:**
1. Check [`experiments/`](experiments/README.md) for evaluation protocols
2. Review [`meta_models_training/`](meta_models_training/README.md) for meta-learning details
3. Explore [`datasets/`](datasets/Dataset_Download_Instructions.md) for data preparation

**For Developers:**
1. Start with [`package/src/smartcal/`](package/src/smartcal/README.md) for API reference
2. Review [`tests/`](tests/README.md) for testing guidelines
3. Check [`experiment_manager/`](experiment_manager/README.md) for database integration

---

> **Note:** All paths are relative to the repository root  
> **Tip:** Click any link to view detailed component documentation  
> **Quick Start:** Run `python package/examples/smartcal_example.py` to get started immediately

---

## Collaborators

This project is a collaborative effort developed with support from **Giza Systems**. We would like to acknowledge the following contributors:

### Core Team
- **Mohamed Maher** - [GitHub](https://github.com/mmaher22) | [PyPI](https://pypi.org/user/mmaher22/) | [Email](mailto:m.maher525@gmail.com)
- **Osama Fayez** - [GitHub](https://github.com/osamaoun97) | [PyPI](https://pypi.org/user/osamaoun97/) | [Email](mailto:osamaoun997@gmail.com)
- **Youssef Medhat** - [GitHub](https://github.com/yossfmedhat) | [PyPI](https://pypi.org/user/yossfmedhat/) | [Email](mailto:yossfmedhat@gmail.com)
- **Mariam Elseedawy** - [GitHub](https://github.com/Seedawy200) | [PyPI](https://pypi.org/user/Seedawy200/) | [Email](mailto:mariam.elseedawy@gmail.com)
- **Yara Marei** - [GitHub](https://github.com/yaramostafa) | [PyPI](https://pypi.org/user/yaramostafa/) | [Email](mailto:yaramostafa500@gmail.com)
- **Abdullah Ibrahim** - [GitHub](https://github.com/BidoS) | [PyPI](https://pypi.org/user/abdullah.ibrahim/) | [Email](mailto:abdullahibrahim544@gmail.com)

### Company Support
- **Giza Systems** - Research support for the development of smartcal

---

## Citation

If you use smartcal in your research or publication, please cite it as:

**Text Citation:**
```text
Maher, M., Fayez, O., Medhat, Y., Elseedawy, M., Marei, Y., & Ibrahim, A. (2025). 
smartcal: A Meta-Learning Approach for Automatic Post-Hoc Calibration of Machine Learning Models. 
GitHub. https://github.com/giza-data-team/SmartCal
```

**BibTeX:**
```bibtex
@software{smartcal2025,
  title={smartcal: A Meta-Learning Approach for Automatic Post-Hoc Calibration of Machine Learning Models},
  author={Maher, Mohamed and Fayez, Osama and Medhat, Youssef and Elseedawy, Mariam and Marei, Yara and Ibrahim, Abdullah},
  year={2025},
  publisher={GitHub},
  organization={Giza Systems},
  url={https://github.com/giza-data-team/SmartCal},
  version={0.1.14}
}
```

**APA Style:**
```text
Maher, M., Fayez, O., Medhat, Y., Elseedawy, M., Marei, Y., & Ibrahim, A. (2025). 
smartcal: A meta-learning approach for automatic post-hoc calibration of machine learning models 
(Version 0.1.14) [Computer software]. GitHub. https://github.com/giza-data-team/SmartCal
```

---

## License

MIT License. See [LICENSE](LICENSE) file for details.