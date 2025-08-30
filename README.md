# HEP-ML-Templates

A **modular, plug-and-play machine learning framework** designed specifically for **High Energy Physics (HEP)** research. Build, test, and deploy ML models with true modularity - swap datasets, models, and preprocessing components with minimal code changes and zero vendor lock-in.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status:** Production-ready with comprehensive validation, beginner-tested setup (100% success rate, <10s per model), and real-world integration case studies demonstrating 3-line dataset swaps.

---

## ✨ Key Features

- 🧩 **True Modularity**: Mix and match components - datasets, models, preprocessing - independently with consistent APIs
- 🎯 **HEP-Optimized**: Built specifically for particle physics data and workflows, including HIGGS benchmark integration
- ⚡ **Rapid Prototyping**: Swap models/datasets with single CLI commands; beginner-tested setup averaging under 10 seconds per model
- 🔧 **Selective Installation**: Install only the components you need via curated "extras" system with preview and validation
- 🚀 **Dual CLI Interface**: Embedded `mlpipe` commands + optional standalone `mlpipe-manager` for flexibility
- 📊 **Standalone Projects**: Export templates to create self-contained, editable ML projects with no repository dependency
- 🤖 **Multi-Algorithm Support**: Traditional ML (XGBoost, Decision Trees, SVM) + Neural Networks (PyTorch, GNNs, Autoencoders)
- 📈 **Advanced Data Splitting**: Train/val/test splits with stratification, time-series support, and reproducible seeding

---

## 🏗️ Core Architecture

HEP-ML-Templates is built around four fundamental concepts:

### 1. **Blocks** - Modular Components
Self-contained Python classes with consistent APIs that hide library-specific details:

```python
from mlpipe.core.registry import register
from mlpipe.core.interfaces import ModelBlock

@register("model.decision_tree")
class DecisionTreeModel(ModelBlock):
    def build(self, config): ...
    def fit(self, X, y): ...
    def predict(self, X): ...
```

### 2. **Registry** - Discovery System
Unified discovery mechanism allowing code and configs to refer to blocks by name:

```yaml
# configs/model/decision_tree.yaml
block: model.decision_tree
max_depth: 10
criterion: gini
random_state: 42
```

### 3. **Configuration-First** - Reproducible Experiments
YAML-driven workflows with CLI overrides keep code stable while you iterate:

```bash
# Swap components at runtime
mlpipe run --overrides model=xgb_classifier data=higgs_uci
mlpipe run --overrides model.params.max_depth=8 preprocessing=data_split
```

### 4. **Extras System** - Selective Installation
Curated package sets map to concrete file collections with discovery, validation, and preview:

```bash
mlpipe list-extras                    # Discover available components
mlpipe extra-details model-xgb        # Inspect what's included
mlpipe preview-install model-xgb evaluation  # Preview before installing
mlpipe install-local model-xgb evaluation --target-dir ./my-project
```

---

## 🚀 Quick Start (30 seconds)

```bash
# 1) Clone & install the core library
git clone https://github.com/Arvind-t33/hep-ml-templates.git
cd hep-ml-templates
pip install -e .

# 2) Discover available components
mlpipe list-extras

# 3) Create a project with XGBoost + evaluation + HIGGS data
mlpipe install-local model-xgb evaluation data-higgs --target-dir ./my-hep-project
cd ./my-hep-project && pip install -e .

# 4) Run the pipeline (components are configurable)
mlpipe run --overrides model=xgb_classifier data=higgs_uci
```

**Alternative manager-style interface:**
```bash
mlpipe-manager list
mlpipe-manager details model-xgb
mlpipe-manager install model-xgb ./my-project
```

---

## 📊 Available Components (Blocks)

> Use `mlpipe list-extras` and `mlpipe extra-details <name>` for exact identifiers and installation details.

### 🎯 **Complete Pipelines** (4)
End-to-end workflows with model + preprocessing + evaluation:
- `pipeline-xgb` (5 blocks, 8 configs) - XGBoost pipeline with preprocessing and metrics
- `pipeline-decision-tree` (5 blocks, 8 configs) - Decision tree complete workflow 
- `pipeline-torch` (4 blocks, 6 configs) - PyTorch neural network pipeline
- `pipeline-gnn` (4 blocks, 6 configs) - Graph neural network pipeline

### 🧠 **Individual Models** (11)
Single algorithms with unified interfaces:

**Traditional ML:**
- `model-decision-tree` (1 blocks, 1 configs), `model-random-forest` (1 blocks, 1 configs), `model-svm` (1 blocks, 1 configs)
- `model-xgb` (1 blocks, 1 configs), `model-mlp` (1 blocks, 1 configs), `model-adaboost` (1 blocks, 1 configs), `model-ensemble` (1 blocks, 1 configs)

**Neural & Graph Models:**
- `model-torch` (1 blocks, 3 configs) (PyTorch neural networks)
- `model-cnn` (1 blocks, 1 configs) (Convolutional networks)
- `model-gnn` (1 blocks, 3 configs) (Graph neural networks via PyTorch Geometric)
- `model-transformer` (1 blocks, 1 configs) (Transformer architectures)

### ⚡ **Algorithm Combos** (9)
Model + preprocessing bundles for quick setup:
- `xgb` (2 blocks, 2 configs), `decision-tree` (2 blocks, 2 configs), `random-forest` (2 blocks, 2 configs), `svm` (2 blocks, 2 configs), `mlp` (2 blocks, 2 configs)
- `ensemble` (2 blocks, 2 configs), `torch` (2 blocks, 2 configs), `gnn` (2 blocks, 2 configs), `adaboost` (2 blocks, 2 configs)

### 📊 **Data Sources** (3)
- `data-higgs` (1 blocks, 1 configs) - HIGGS benchmark dataset (validated integration)
- `data-csv` (1 blocks, 1 configs) - Universal CSV loader with flexible configuration
- `data-split` (1 blocks, 1 configs) - Advanced train/val/test splitting utilities

### 🏗️ **Component Categories** (3)
- `preprocessing` (3 blocks, 2 configs) - Scaling, feature engineering, data splitting
- `evaluation` (2 blocks, 2 configs) - Classification metrics, reconstruction metrics
- `feature-eng` (1 blocks, 2 configs) - Feature engineering demonstrations

### 🌟 **Special** (1)
- `all` (16 blocks, 27 configs) - Complete bundle

---

## 🛠️ Three Core Workflows

### 1. **Rapid Prototyping**
Experiment with different models and datasets using config/CLI overrides:

```bash
# Try different models on the same data
mlpipe run --overrides model=decision_tree
mlpipe run --overrides model=xgb_classifier
mlpipe run --overrides model=random_forest

# Switch datasets and preprocessing
mlpipe run --overrides data=csv_demo preprocessing=time_series_split
mlpipe run --overrides data=higgs_100k feature_eng=demo_features
```

### 2. **Standalone Project Scaffolding** 
Create self-contained projects with selected components:

```bash
# Create a new project directory
mlpipe install-local model-random-forest data-higgs evaluation --target-dir ./research-project
cd ./research-project && pip install -e .

# Add more components later
mlpipe install-local model-xgb preprocessing .
mlpipe run --overrides model=xgb_classifier preprocessing=stratified_split
```

### 3. **Integration into Existing Code**
Drop in individual blocks with minimal changes (~3 lines):

**Before (traditional scikit-learn):**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict_proba(X_test_scaled)[:, 1]
```

**After (with hep-ml-templates):**
```python
from mlpipe.blocks.model.ensemble_models import RandomForestModel  # Change 1

config = {'n_estimators': 100, 'random_state': 42}
model = RandomForestModel()                                        # Change 2
model.build(config)
model.fit(X_train, y_train)                                        # Change 3 - preprocessing handled internally
predictions = model.predict_proba(X_test)[:, 1]
```

**Swap to XGBoost:**
```python
from mlpipe.blocks.model.xgb_classifier import XGBClassifierModel  # Only import changes
model = XGBClassifierModel()                                       # Only class name changes
model.build({'n_estimators': 200, 'learning_rate': 0.1})
```

---

## 🔄 Advanced Data Splitting

Built-in splitting utilities with comprehensive support:

### **Convenience Function:**
```python
from mlpipe.blocks.preprocessing.data_split import split_data

splits = split_data(X, y, 
    train_size=0.7, val_size=0.15, test_size=0.15, 
    stratify=True, random_state=42
)
X_train, y_train = splits['train']
X_val, y_val = splits['val']
X_test, y_test = splits['test']
```

### **Class-Based Approach:**
```python
from mlpipe.blocks.preprocessing.data_split import DataSplitter

splitter = DataSplitter({
    'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
    'stratify': True, 'time_series': False, 'random_state': 42
})
splits = splitter.fit_transform(X, y)
```

### **Pipeline Integration:**
```bash
# Use pre-configured processing strategies
mlpipe run --overrides preprocessing=data_split
mlpipe run --overrides preprocessing=standard
mlpipe run --overrides feature_eng=column_selector
```

### **Configuration Examples:**

**Stratified 70/15/15 Split:**
```yaml
# configs/preprocessing/data_split.yaml
train_size: 0.7
val_size: 0.15
test_size: 0.15
stratify: true
shuffle: true
random_state: 42
```

**Standard Preprocessing:**
```yaml
# configs/preprocessing/standard.yaml
with_mean: true
with_std: true
copy: true
```

---

## 💻 Complete CLI Reference

### **Embedded CLI (`mlpipe`)**

#### **Discovery & Configuration Commands**
```bash
# List all available blocks (registered components)
mlpipe list-blocks

# List all available configurations with usage examples
mlpipe list-configs [--config-path CONFIGS_DIR]

# Discover available extras and their contents
mlpipe list-extras                                      # Show all available extras by category

# Inspect specific extras before installing
mlpipe extra-details EXTRA_NAME                        # Show detailed breakdown of blocks/configs
mlpipe extra-details model-xgb                         # Example: inspect XGBoost extra

# Preview installations before committing
mlpipe preview-install EXTRA1 [EXTRA2 ...]            # Preview what would be installed
mlpipe preview-install model-xgb evaluation            # Example: preview installation

# Validate the extras system integrity
mlpipe validate-extras                                  # Check all extras mappings are valid
```

#### **Installation & Setup Commands**
```bash
# Install extras locally to create standalone projects
mlpipe install-local EXTRA1 [EXTRA2 ...] --target-dir TARGET_DIR

# Examples:
mlpipe install-local model-xgb --target-dir ./my-xgb-project
mlpipe install-local model-decision-tree data-higgs evaluation --target-dir ./research-project
mlpipe install-local all --target-dir ./complete-ml-suite
```

#### **Execution & Pipeline Commands**
```bash
# Run pipelines with full configuration control
mlpipe run [OPTIONS]

# Pipeline options:
mlpipe run                                              # Use defaults (xgb_basic pipeline)
mlpipe run --pipeline PIPELINE_NAME                    # Specify pipeline implementation
mlpipe run --config-path CONFIGS_DIR                   # Custom config directory
mlpipe run --config-name CONFIG_FILE                   # Specific pipeline config file

# Override any configuration values:
mlpipe run --overrides OVERRIDE1 [OVERRIDE2 ...]
mlpipe run --overrides model=xgb_classifier            # Swap model component
mlpipe run --overrides data=higgs_uci                  # Swap data component
mlpipe run --overrides model=decision_tree data=csv_demo  # Multiple overrides

# Parameter-level overrides (dot notation):
mlpipe run --overrides model.params.max_depth=8        # Model hyperparameters
mlpipe run --overrides model.params.n_estimators=200 model.params.learning_rate=0.1
mlpipe run --overrides data.params.test_size=0.2       # Data splitting parameters
```

### **Manager CLI (`mlpipe-manager`)**
Standalone interface with simpler command structure:

```bash
# Discovery commands
mlpipe-manager list                                     # List all available extras
mlpipe-manager validate                                 # Validate extras configuration

# Inspection commands  
mlpipe-manager details EXTRA_NAME                      # Show details for specific extra
mlpipe-manager preview EXTRA1 [EXTRA2 ...]            # Preview installation

# Installation command
mlpipe-manager install EXTRA1 [EXTRA2 ...] TARGET_DIR  # Install extras to directory

# Examples:
mlpipe-manager details model-xgb                       # Inspect XGBoost extra
mlpipe-manager preview model-xgb preprocessing         # Preview combined installation
mlpipe-manager install model-xgb ./my-project          # Install to project directory
```

### **Complete Usage Examples**

#### **Basic Model Training**
```bash
# Quick start with defaults
mlpipe run

# Try different models on same data
mlpipe run --overrides model=decision_tree
mlpipe run --overrides model=random_forest
mlpipe run --overrides model=svm

# Switch datasets
mlpipe run --overrides data=csv_demo
mlpipe run --overrides data=higgs_uci
```

#### **Hyperparameter Tuning**
```bash
# XGBoost hyperparameter sweep
mlpipe run --overrides model=xgb_classifier model.params.max_depth=6
mlpipe run --overrides model=xgb_classifier model.params.max_depth=8
mlpipe run --overrides model=xgb_classifier model.params.n_estimators=200 model.params.learning_rate=0.05

# Decision tree parameters
mlpipe run --overrides model=decision_tree model.params.max_depth=10 model.params.min_samples_split=5
```

#### **Data Processing Variations**
```bash
# Different preprocessing strategies  
mlpipe run --overrides preprocessing=standard          # Standard scaling
mlpipe run --overrides preprocessing=data_split        # Custom data splitting

# Combined data and preprocessing changes
mlpipe run --overrides data=higgs_uci preprocessing=standard model=xgb_classifier
```

#### **Project Creation Workflow**
```bash
# 1. Explore available components
mlpipe list-extras
mlpipe extra-details pipeline-xgb

# 2. Preview what will be installed
mlpipe preview-install pipeline-xgb

# 3. Create project with selected components
mlpipe install-local pipeline-xgb --target-dir ./hep-research
cd ./hep-research && pip install -e .

# 4. Run experiments with different configurations
mlpipe run --overrides model.params.max_depth=8
mlpipe run --overrides data=csv_demo
```

---

## ⚙️ Installation & Dependency Management

### **Development Installation**
```bash
# Full development setup with all dependencies
git clone https://github.com/Arvind-t33/hep-ml-templates.git
cd hep-ml-templates
pip install -e '.[all]'
```

### **Selective Installation**
Install only the dependencies you need:

```bash
# Core framework only
pip install -e '.[core]'

# Traditional ML models
pip install -e '.[xgb,decision-tree,random-forest,svm]'

# Deep learning components
pip install -e '.[torch,gnn,autoencoder]'

# Data science essentials
pip install -e '.[data-csv,data-higgs,preprocessing,evaluation]'
```

### **Available Extras in `pyproject.toml`:**
- **Individual Components:** `model-xgb`, `model-decision-tree`, `model-torch`, etc.
- **Algorithm Groups:** `xgb`, `torch`, `gnn`, `ensemble`
- **Data & Processing:** `data-csv`, `data-higgs`, `preprocessing`, `evaluation`
- **Complete Bundle:** `all` (includes everything)

---

## 📁 Project Structure

```
hep-ml-templates/
├── src/mlpipe/                     # Core library source
│   ├── blocks/                     # Modular components
│   │   ├── model/                  # ML models (traditional + neural)
│   │   ├── ingest/                 # Data loading (CSV, HIGGS, etc.)
│   │   ├── preprocessing/          # Data splitting, scaling, feature eng
│   │   ├── evaluation/             # Metrics and evaluation blocks
│   │   └── training/               # Training orchestration
│   ├── core/                       # Framework interfaces & registry
│   │   ├── interfaces.py           # Base block interfaces
│   │   ├── registry.py             # Component discovery system
│   │   ├── config.py               # Configuration management
│   │   └── utils.py                # Utility functions
│   └── cli/                        # Command-line interfaces
│       ├── main.py                 # `mlpipe` commands
│       ├── manager.py              # `mlpipe-manager` (standalone)
│       └── local_install.py        # Extras installation logic
├── configs/                        # Default YAML configurations
│   ├── model/                      # Model configurations
│   ├── data/                       # Data loader configurations  
│   ├── preprocessing/              # Preprocessing configurations
│   └── pipeline/                   # End-to-end pipeline configurations
├── comprehensive_documentation/    # Complete documentation hub
├── tests/                          # Test suites (unit + integration)
├── pyproject.toml                  # Project metadata, dependencies, CLI entry points
└── README.md                       # This file
```

---

---

## 🗂️ Complete Component Reference

### **Available Blocks** (`mlpipe list-blocks`)
```
eval.classification          # Classification evaluation metrics
feature.column_selector      # Feature selection utilities
ingest.csv                   # CSV data loading
model.adaboost              # AdaBoost classifier
model.ae_vanilla            # Vanilla autoencoder
model.ae_variational        # Variational autoencoder
model.cnn_hep               # Convolutional neural network
model.decision_tree         # Decision tree classifier
model.ensemble_voting       # Voting ensemble classifier
model.mlp                   # Multi-layer perceptron
model.random_forest         # Random forest classifier
model.svm                   # Support vector machine
model.transformer_hep       # Transformer architecture
model.xgb_classifier        # XGBoost classifier
preprocessing.data_split    # Data splitting utilities
preprocessing.standard_scaler # Standard scaling preprocessing  
train.sklearn               # Scikit-learn training orchestration
```

### **Available Configurations** (`mlpipe list-configs`)

**Pipeline Configurations:**
- `pipeline` - Default end-to-end pipeline

**Data Configurations:**
- `csv_demo` - Demo CSV dataset configuration
- `custom_hep_example` - Custom HEP dataset example
- `custom_test` - Custom test dataset
- `higgs_uci` - HIGGS UCI dataset configuration
- `medical_example` - Medical dataset example
- `wine_quality_example` - Wine quality dataset example

**Model Configurations:**
- `adaboost` - AdaBoost classifier settings
- `ae_lightning` - Lightning autoencoder settings
- `ae_vanilla` - Vanilla autoencoder settings
- `ae_variational` - Variational autoencoder settings
- `cnn_hep` - CNN for HEP data settings
- `decision_tree` - Decision tree parameters
- `ensemble_voting` - Voting ensemble settings
- `gnn_gat` - Graph Attention Network settings
- `gnn_gcn` - Graph Convolutional Network settings
- `gnn_pyg` - PyTorch Geometric GNN settings
- `mlp` - Multi-layer perceptron settings
- `random_forest` - Random forest parameters
- `svm` - SVM classifier settings
- `transformer_hep` - Transformer for HEP settings
- `xgb_classifier` - XGBoost classifier parameters

**Preprocessing Configurations:**
- `data_split` - Data splitting parameters
- `standard` - Standard scaling parameters

**Feature Engineering Configurations:**
- `column_selector` - Column selection settings
- `custom_test_features` - Custom test features
- `demo_features` - Demo feature engineering

**Training Configurations:**
- `sklearn` - Scikit-learn training parameters

**Evaluation Configurations:**
- `classification` - Classification evaluation metrics

## 🧪 Validation & Testing

### **Comprehensive Validation Results**
- ✅ **6 Core Models Tested:** Decision Tree, Random Forest, XGBoost, SVM, MLP, Ensemble Voting
- ✅ **100% Success Rate:** All models working across different environments
- ✅ **Beginner Testing:** Average setup time <10 seconds per model, rated "extremely easy"
- ✅ **Real-World Integration:** HIGGS benchmark integrated with only 3 line changes
- ✅ **Extras System:** Comprehensive validation across 29 extras with preview/install/validate functionality

### **Production Readiness Indicators**
- 🔍 **Comprehensive Test Suite:** Unit tests, integration tests, end-to-end validation
- 📚 **Complete Documentation:** Master documentation index with guides, reports, and case studies
- 🌐 **Real-World Case Study:** HIGGS100K dataset integration demonstrates practical applicability
- 🔧 **Robust Installation:** Local installation system with dependency management and validation
- ⚡ **Performance Verified:** All models produce expected training/evaluation outputs

---

## 🤝 Contributing

We welcome contributions of new models, datasets, preprocessing utilities, evaluation blocks, and documentation.

### **Adding a New Model**

1. **Implement the Model:**
```python
from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register

@register("model.my_new_model")
class MyNewModel(ModelBlock):
    def build(self, config):
        # Initialize model with config parameters
        pass
    
    def fit(self, X, y):
        # Train the model
        pass
    
    def predict(self, X):
        # Make predictions
        pass
    
    def predict_proba(self, X):  # For classification
        # Return prediction probabilities
        pass
```

2. **Create Configuration:**
```yaml
# configs/model/my_new_model.yaml
block: model.my_new_model
param1: default_value
param2: another_default
random_state: 42
```

3. **Update Extras Mapping:**
Add your model to the extras system in `cli/local_install.py`

4. **Add Tests:**
Create unit tests and integration tests for your model

5. **Update Documentation:**
Add usage examples and update the model list

### **Development Setup**
```bash
git clone https://github.com/Arvind-t33/hep-ml-templates.git
cd hep-ml-templates
pip install -e '.[all]'
# Run tests
python -m pytest tests/ -v
# Validate extras system
mlpipe validate-extras
```

See `CONTRIBUTING.md` for full guidelines, coding standards, and review process.

---

## ❓ FAQ & Troubleshooting

### **Installation Issues**

**Q: Import errors after installation**
```bash
# Ensure you're in the correct directory and installed in editable mode
cd /path/to/your/project
pip install -e .
# Validate the extras system
mlpipe validate-extras
```

**Q: "Model not found" errors**
```bash
# Check what's available
mlpipe list-extras
mlpipe extra-details model-name
# Ensure the model was installed
mlpipe preview-install model-name
```

### **Configuration Questions**

**Q: How do I change hyperparameters without editing YAML files?**
```bash
# Use dotted notation for parameter overrides
mlpipe run --overrides model=xgb_classifier model.params.max_depth=8
mlpipe run --overrides model.params.n_estimators=200 model.params.learning_rate=0.1
```

**Q: How do I combine multiple overrides?**
```bash
# Multiple components and parameters
mlpipe run --overrides data=higgs_uci model=xgb_classifier preprocessing=data_split model.params.max_depth=8
```

### **Development Questions**

**Q: How do I preview what components will be installed?**
```bash
# Preview before installing
mlpipe preview-install model-xgb evaluation data-higgs
# Check specific extra contents
mlpipe extra-details model-xgb
```

**Q: How do I validate my installation?**
```bash
# Validate the entire extras system
mlpipe validate-extras
# Test specific functionality
mlpipe list-blocks
mlpipe list-configs
```

---

## 🏆 Research Impact & Applications

### **High Energy Physics Applications**
- **HIGGS Benchmark Integration:** Demonstrated with 3-line code changes, maintaining 100% existing functionality
- **Multi-Model Comparison:** Easy benchmarking across traditional ML and neural network approaches
- **Reproducible Experiments:** Configuration-driven workflows with explicit seeds and consistent data splitting

### **Research Workflow Benefits**
- **Rapid Prototyping:** Test multiple algorithms on the same dataset in minutes
- **Easy Dataset Switching:** Change from demo data to production HIGGS data with single CLI override
- **Collaborative Research:** Share self-contained projects with consistent APIs across teams
- **Paper-Ready Results:** Comprehensive documentation supports research publication requirements

### **Production Deployment**
- **Modular Architecture:** Deploy only the components needed for specific use cases
- **Version Control Friendly:** Configuration-first approach enables clear experiment tracking
- **Scalable Design:** Add new models, datasets, and preprocessing without breaking changes

---

## 📄 License & Acknowledgments

- **License:** MIT License - see `LICENSE` file for details
- **Built On:** Python scientific stack including scikit-learn, XGBoost, pandas, PyTorch, PyTorch Geometric
- **Supported By:** IRIS-HEP fellowship program
- **Community:** Made possible by the High Energy Physics and machine learning communities

### **Citation**
If you use HEP-ML-Templates in your research, please cite:
```bibtex
@software{hep_ml_templates,
  title={HEP-ML-Templates: A Modular Machine Learning Framework for High Energy Physics},
  author={Tawker, Arvind},
  year={2025},
  url={https://github.com/Arvind-t33/hep-ml-templates},
  note={IRIS-HEP Fellowship Project}
}
```

---

## 🚀 Getting Started Now

Ready to start? Here's your path forward:

### **For Quick Experimentation:**
```bash
git clone https://github.com/Arvind-t33/hep-ml-templates.git
cd hep-ml-templates && pip install -e '.[all]'
mlpipe run --overrides model=xgb_classifier
```

### **For New Projects:**
```bash
# In your project directory
mlpipe install-local model-xgb data-higgs evaluation --target-dir .
pip install -e .
mlpipe run
```

### **For Existing Code Integration:**
```bash
# Install specific components
mlpipe install-local model-random-forest preprocessing --target-dir .
# Update imports (see integration examples above)
```

**Questions?** Check the FAQ above, explore `comprehensive_documentation/`, or open an issue on GitHub.

---

*HEP-ML-Templates: Making machine learning in High Energy Physics modular, reproducible, and accessible.*
