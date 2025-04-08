# SmartCal

## 📋 Project Overview

**SmartCal** is an open-source AutoML framework designed to automate the selection and tuning of post-hoc calibration methods for classification models. It addresses the critical challenge of obtaining accurate probability estimates—essential for high-stakes domains like medical diagnosis, fraud detection, and autonomous systems—by dynamically recommending the best calibration strategy from a pool of **12 methods** based on dataset characteristics and classifier behavior.

### 🎯 Goals

- **Automate Calibration**: Streamline the calibration process of classification models to improve reliability with minimal user intervention.
- **Support Multiple Domains**: Enable calibration for a wide range of datasets, including those from image, language, and tabular domains.
- **Provide Calibration Metrics**: Evaluate calibration methods using metrics like Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Confidence ECE, and Execution Time Overhead.
- **Meta-Learning**: Develop a meta-model that recommends the best calibration method based on dataset characteristics (e.g., kurtosis, skewness, histogram binning).

## ⭐ Key Features

- **Automated Calibration Selection**: Meta-model trained on **172 datasets** (tabular/image/text) and **13 classifiers** recommends optimal methods.
- **Unified Interface**: Single API for 12 calibration techniques (Platt Scaling, Temperature Scaling, Beta Calibration, Isotonic Regression, etc.).
- **Multi-Domain Support**: Works with binary/multi-class tasks across tabular, image, and language data.
- **Bayesian Optimization**: Fine-tunes hyperparameters for selected calibrators, outperforming random search.
- **Comprehensive Metrics**: Evaluates performance using:
  - Expected Calibration Error (ECE) 
  - Maximum Calibration Error (MCE)  
  - Brier Score  
  - Calibration Curves  
  - Execution Time Overhead  
  
## 🔄 Methodology

The project follows a systematic approach to build and evaluate calibration algorithms:

1. **Dataset Collection**: Gather datasets across binary and multi-class classification tasks, ensuring a diverse range of classification problems.
2. **Knowledge Base Construction**: Store detailed metadata for each experiment to facilitate later use in the meta-model.
3. **Meta-Model Development**: Train a meta-learning model that suggests the most appropriate calibration method based on dataset characteristics.
4. **Performance Benchmarking**: Evaluate and compare different calibration techniques using defined metrics, and benchmark the meta-model against other search strategies (e.g., random search) for selecting the best method.

---
## 📁 Repository Structure: AUTOML CALIBRATOR
```python
AUTOML CALIBRATOR/
├── Package/src/SmartCal/      # Main package directory
│   ├── autocal_end_to_end/      # End-to-end calibration implementation using baysian opt.
│   ├── calibration_algorithms/   # Various calibration method implementations
│   ├── config/                  # Configuration files and settings
│   ├── meta_features_extraction/ # Feature extraction utilities
│   ├── meta_model/              # Meta-learning model implementation
│   ├── metrics/                 # Calibration evaluation metrics
│   ├── SmartCal/               # Core SmartCal implementation
│   ├── utils/                  # Utility functions and helpers
│   └── __init__.py             # Package initialization file
├── baseline_random_search/     # Baseline implementation and random search
├── classifiers/               # Model implementations for different data types
├── data_preparation/          # Data preprocessing and preparation
├── Datasets/                  # Dataset storage and management
├── experiment_manager/        # Experiment tracking and management on databases
├── experiments/              # Experiment configurations and scripts
├── figures/                  # Figures to analyze package benchmarking
├── meta_data_extraction/      # Script for data meta features extraction from our knowledgebase
├── meta_model_training/      # Training meta model
├── knowledge_base/           # Knowledge base script for meta-learning
├── pipeline/                 # Processing pipelines for different data types
├── tests/                   # Test suite
├── docker-compose.yml       # Docker configuration for running the PostgreSQL database and pgAdmin.
└── requirements.txt         # Python packages and dependencies required to run the project or test suite.
```

### Key Components Description

#### `Package/src/SmartCal/`
- Core package containing the main implementation
- Houses the SmartCal algorithm and supporting modules 
- Contains all calibration algorithms, meta-features extractor, meta-model and metrics

#### `baseline_random_search/`
- Implementation of baseline methods
- Random search algorithms for comparison

#### `classifiers/`
- Various classifier implementations
- Support for different data types (image, language, tabular)

#### `data_preparation/`
- Data info for different types of data
- Data splitting and preprocessing pipelines

#### `Datasets/`
- Dataset storage and management
- Data loading utilities

#### `experiment_manager/`
- Experiment tracking functionality
- Results saving and management on databases

#### `experiments/`
- Experiment configurations
- Test scripts and evaluation protocols

#### `figures/`
- Scripts to generate benchmarking analysis visualizations

#### `knowledge_base/`
- Scripts for running experiments to generate knowledge base

#### `meta_data_extraction/`
- Script to extract meta features from our knowledge base

#### `meta_model_training/`
- Script to train One vs. All meta model

#### `pipeline/`
- End-to-end pipelines for automated calibration and hyperparameter optimization

#### `tests/`
- Unit tests for all components 

---

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.12.7
- Docker
- PostgreSQL

### 1️⃣ Environment Setup

First, clone the repository and create a virtual environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/AutoML Calibration.git
cd AutoML Calibration

# Create virtual environment (choose one)
# Using conda
conda create -n autocal python=3.12

# OR using venv
python3.12 -m venv autocal

# Activate the environment
# For conda
conda activate autocal

# For venv
# Windows
.\autocal\Scripts\activate
# Unix/MacOS
source autocal/bin/activate

# Install requirements
pip install -r requirements.txt

# Run this command
$env:PYTHONPATH="Package/src"
```
### 2️⃣ Quick Start
To verify your installation, run the example script:

```bash
python Package\Examples\SmartCal_Example.py
```
### 3️⃣ Database Setup 
SmartCal uses PostgreSQL to store experiment results and meta-learning data. Follow these steps to set up the database:

- 🐳 Start PostgreSQL Container
```bash
# Start PostgreSQL using Docker
docker-compose up -d
```
- ⚙️ Configure Database Credentials
Database configurations such as port number, host, username, and password are located in:
(`config/resources/conf/config-dev.env`)

Make sure these match your Alembic and application settings and SSH_ENABLED=false and DB_PASSWORD = password in (`config/resources/conf/config-dev.env`).

- 🗄️ Initialize Database Schema
```bash
# Create migrations directory
mkdir -p experiment_manager/migrations/versions

# Generate initial migration
alembic -c experiment_manager/alembic.ini revision --autogenerate -m "Initial migration"

# Apply migrations
alembic -c experiment_manager/alembic.ini upgrade head
```
This will apply all migrations and create your database tables.

- 🔍 Verify Setup (Optional)
For WSL users or to verify the database:

```bash
# Connect to PostgreSQL
psql -h localhost -U admin -d Calibration_db -p 5432

# Common psql commands:
# \dt - List tables
# \d+ table_name - Describe table
# \q - Exit psql
```

## 📚 Documentation for Each Component

### Core Components
- - [`Package/`](Package/README.md) - Package documentation
- [`Package/src/SmartCal/calibration_algorithms/`](Package/src/SmartCal/calibration_algorithms/README.md) - Implementation of 12 calibration methods  
- [`Package/src/SmartCal/meta_features_extraction/`](Package/src/SmartCal/meta_features_extraction/README.md) - Feature extraction utilities  
- [`Package/src/SmartCal/SmartCal/`](Package/src/SmartCal/SmartCal/README.md) - Core calibration engine  

### Data & Experiments
- [`Datasets/`](Datasets/Dataset_Download_Instructions.md) - Dataset download and preparation  
- [`data_preparation/preprocessors`](data_preparation/preprocessors/README.md) - Datasets preprocessing
- [`experiments/end_to_end_eval/`](experiments/end_to_end_eval/README.md) - Full pipeline evaluation  
- [`experiments/meta_model_eval/`](experiments/meta_model_eval/README.md) - Meta-model benchmarking  
- [`figures/`](figures/README.md) -  Generating and storing visualizations.
- [`meta_data_extraction/`](meta_data_extraction/README.md) - Meta-features extraction from DB
- [`meta_model_training/`](meta_model_training/README.md) - Meta-Model Construction

### Knowledge System
- [`knowledge_base/`](knowledge_base/README.md) - Experimental results database  

> 🔗 *All paths are relative to the repository root*  
> 💡 *Click any link to view component documentation*