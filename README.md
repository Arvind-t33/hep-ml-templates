# HEP ML Templates 🚀

**The modern, modular machine learning framework for High Energy Physics**

Built by researchers, for researchers. Get from raw data to trained models in minutes, not hours.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![HEP](https://img.shields.io/badge/domain-High%20Energy%20Physics-purple.svg)

---

## ✨ What Makes This Different?

🔧 **True Modularity**: Mix and match any dataset, model, or preprocessing step with a single command  
📦 **Smart Installation**: Install only what you need with pip extras  
🎯 **Researcher-Focused**: Built specifically for HEP workflows and datasets  
⚡ **Zero Config**: Sensible defaults get you started immediately  
🌐 **Auto-Download**: HIGGS dataset downloads automatically when needed  
🏠 **Local Control**: Install blocks locally for full customization  

---

## 🚀 Quick Start (30 seconds)

### Option 1: Complete Pipeline (Recommended for beginners)
```bash
# Install complete XGBoost pipeline with HIGGS dataset
pip install 'hep-ml-templates[pipeline-xgb]'

# Run your first ML experiment (downloads HIGGS dataset automatically)
mlpipe run
```

### Option 2: Pick Your Components (For advanced users)
```bash
# Install only what you need
pip install 'hep-ml-templates[model-xgb,data-higgs,preprocessing]'

# Or install locally for full control
mlpipe install-local pipeline-xgb --blocks-dir my_project_blocks
```

**That's it!** 🎉 Your first HEP ML pipeline is running.

---

## 🎯 For New Researchers: Start Here

### 1️⃣ **Your First Experiment** (5 minutes)

Start with the HIGGS dataset - a classic HEP binary classification problem:

```bash
# Install and run - everything is automatic
pip install 'hep-ml-templates[pipeline-xgb]'
mlpipe run

# See available components
mlpipe list-configs
mlpipe list-blocks
```

**What just happened?**
- ✅ HIGGS dataset (100k samples) downloaded automatically
- ✅ Data preprocessed with physics-aware defaults  
- ✅ XGBoost model trained and evaluated
- ✅ Results saved with performance metrics

### 2️⃣ **Try Different Models** (1 minute)

Switch algorithms effortlessly:

```bash
# Try different model architectures
mlpipe run --overrides model=neural_network
mlpipe run --overrides model=xgb_classifier

# Mix with different preprocessing
mlpipe run --overrides model=xgb_classifier preprocessing=hepphys
```

### 3️⃣ **Use Your Own Data** (5 minutes)

Got your own CSV dataset? No problem:

```python
from mlpipe.core.registry import get
import mlpipe.blocks

# Load any CSV automatically
loader = get('ingest.csv')({
    'file_path': 'my_data.csv',
    'target_column': 'signal',
    'has_header': True
})
X, y, metadata = loader.load()

# Use any model from the registry
model = get('model.xgb_classifier')()
model.fit(X, y)
predictions = model.predict(X)
```

---

## 🔧 Installation Options

### Smart Pip Extras - Install Only What You Need

```bash
# 🎯 Core framework only (minimal)
pip install hep-ml-templates

# 📊 Individual components
pip install 'hep-ml-templates[model-xgb]'      # Just XGBoost
pip install 'hep-ml-templates[data-higgs]'     # Just HIGGS dataset  
pip install 'hep-ml-templates[preprocessing]'  # Just preprocessing tools

# 🚀 Combined workflows (most popular)
pip install 'hep-ml-templates[model-xgb,data-higgs]'
pip install 'hep-ml-templates[xgb]'            # XGBoost + preprocessing
pip install 'hep-ml-templates[torch]'          # PyTorch + Lightning

# 🏆 Complete pipelines (everything included)
pip install 'hep-ml-templates[pipeline-xgb]'   # Full XGBoost pipeline
pip install 'hep-ml-templates[pipeline-torch]' # Full PyTorch pipeline
pip install 'hep-ml-templates[all]'            # Everything available
```

### Local Installation - Full Control

Perfect for researchers who want to modify code locally:

```bash
# Install complete pipeline locally
mlpipe install-local pipeline-xgb --blocks-dir my_research_project

# Install specific components only  
mlpipe install-local data-higgs model-xgb --blocks-dir my_project

# Add more components later
mlpipe install-local preprocessing --blocks-dir my_project
```

**What you get locally:**
- 📁 Complete source code for installed blocks
- ⚙️ All configuration files
- 🔧 Full customization control
- 📝 No internet dependency after installation

---

## 📚 Tutorials & Examples

### Tutorial 1: Quick Experimentation

**Goal**: Try different model/data combinations rapidly

```bash
# Start with defaults
mlpipe run

# Switch datasets instantly
mlpipe run --overrides data=csv_demo
mlpipe run --overrides data=higgs_uci

# Try different models
mlpipe run --overrides model=xgb_classifier
mlpipe run --overrides model=neural_network

# Combine any components
mlpipe run --overrides data=csv_demo model=xgb_classifier preprocessing=standard
```

### Tutorial 2: Your Own Dataset Integration

**Goal**: Use your CSV data with the framework

**Step 1**: Create dataset configuration (`configs/data/my_experiment.yaml`):
```yaml
block: ingest.csv
file_path: "data/my_detector_data.csv"
target_column: "is_signal"
has_header: true
nrows: 50000  # Limit for testing
```

**Step 2**: Update pipeline configuration (`configs/pipeline.yaml`):
```yaml
data: my_experiment  # Points to your config
preprocessing: standard
model: xgb_classifier
# ... rest unchanged
```

**Step 3**: Run your pipeline:
```bash
mlpipe run  # Uses your dataset automatically
```

### Tutorial 3: Model Comparison Study

**Goal**: Compare multiple models on the same dataset

```bash
# Create a systematic comparison
mlpipe run --overrides model=xgb_classifier > results_xgb.txt
mlpipe run --overrides model=neural_network > results_nn.txt

# Or use Python for programmatic comparison
python -c "
from mlpipe.core.registry import get
import mlpipe.blocks

# Load data once
loader = get('data.higgs')()
X, y, _ = loader.load()

# Compare models
models = ['xgb_classifier', 'neural_network']
for model_name in models:
    model = get(f'model.{model_name}')()
    model.fit(X, y)
    score = model.score(X, y)
    print(f'{model_name}: {score:.4f}')
"
```

### Tutorial 4: Local Development Workflow

**Goal**: Modify and extend the framework locally

```bash
# 1. Install locally for development
mlpipe install-local pipeline-xgb --blocks-dir my_research

# 2. Your project structure
my_research/
├── blocks/           # ← Modify these
│   ├── model/
│   ├── preprocessing/
│   └── ingest/
├── configs/          # ← Customize these
│   ├── data/
│   └── model/
└── core/            # ← Framework internals

# 3. Customize and run
cd my_research
python -c "
import sys
sys.path.insert(0, '.')  # Use local blocks

from blocks.model.xgb_classifier import XGBClassifierBlock
# Modify the class, add features, etc.
"
```

---

## 🏗️ Architecture & Components

### Core Philosophy: True Modularity

Every component is **swappable** and **independent**:

```python
# Registry system - everything is pluggable
from mlpipe.core.registry import get

# Any data source
data_loader = get('ingest.csv')        # CSV files
data_loader = get('data.higgs')        # HIGGS dataset 
data_loader = get('ingest.root')       # ROOT files (coming soon)

# Any preprocessing
preprocessor = get('preprocessing.standard_scaler')  # Standard scaling
preprocessor = get('preprocessing.hepphys')         # HEP-specific preprocessing

# Any model
model = get('model.xgb_classifier')    # XGBoost
model = get('model.neural_network')    # Neural network
model = get('model.gnn_pyg')          # Graph Neural Network

# Everything works together seamlessly
```

### Component Categories

| Component | Purpose | Examples |
|-----------|---------|----------|
| 📊 **Data Ingest** | Load datasets | `ingest.csv`, `data.higgs` |
| 🔧 **Preprocessing** | Prepare data | `preprocessing.standard_scaler`, `preprocessing.hepphys` |
| ✨ **Feature Engineering** | Feature selection/creation | `feature.column_selector`, `feature.physics_features` |
| 🤖 **Models** | ML algorithms | `model.xgb_classifier`, `model.neural_network`, `model.gnn_pyg` |
| 📈 **Training** | Training procedures | `training.sklearn`, `training.pytorch_lightning` |
| 📊 **Evaluation** | Performance metrics | `eval.classification`, `eval.reconstruction` |

### Configuration System

**Hierarchical and modular** - each component has its own config space:

```
configs/
├── pipeline.yaml          # 🎯 Single control center
├── data/                 # 📊 Dataset configurations
│   ├── higgs_uci.yaml   
│   ├── csv_demo.yaml
│   └── my_dataset.yaml
├── model/                # 🤖 Model configurations  
│   ├── xgb_classifier.yaml
│   ├── neural_network.yaml
│   └── my_model.yaml
├── preprocessing/        # 🔧 Preprocessing configurations
└── ...
```

**Two ways to change components:**
1. **Temporary**: `mlpipe run --overrides data=csv_demo model=xgb_classifier`
2. **Persistent**: Edit `configs/pipeline.yaml` to change defaults

---

## 🎯 Supported Use Cases

### ✅ **New Researcher** 
*"I want to get started quickly"*
```bash
pip install 'hep-ml-templates[pipeline-xgb]'
mlpipe run
```

### ✅ **Experimenter** 
*"I want to try different model/data combinations"*
```bash
mlpipe run --overrides model=neural_network
mlpipe run --overrides data=csv_demo model=xgb_classifier
```

### ✅ **Data Scientist** 
*"I have my own dataset"*
- Built-in CSV loader with auto-detection
- Easy configuration for any tabular data
- Seamless integration with existing models

### ✅ **Model Developer** 
*"I want to add my own model"*
- Clear interfaces for new models
- Automatic registry integration
- Local installation for development

### ✅ **HEP Researcher** 
*"I need HEP-specific tools"*
- HIGGS dataset with auto-download
- Physics-aware preprocessing 
- HEP-optimized defaults

---

## 📦 Datasets & Models

### Built-in Datasets

| Dataset | Description | Size | Auto-Download |
|---------|-------------|------|---------------|
| **HIGGS** | UCI HEP binary classification | 100k samples, 28 features | ✅ |
| **CSV Demo** | Example tabular data | 1k samples | ✅ |
| **Custom CSV** | Your own data | Any size | ➖ |

### Built-in Models

| Model | Type | Use Case | Dependencies |
|-------|------|----------|-------------|
| **XGBoost** | Tree ensemble | Classification/Regression | `xgboost` |
| **Neural Network** | Deep learning | General purpose | `pytorch`, `lightning` |
| **GNN** | Graph neural network | Graph data | `torch-geometric` |

### Preprocessing Tools

- **Standard Scaler**: Z-score normalization
- **HEP Physics**: Physics-aware preprocessing
- **Column Selector**: Feature selection
- **Missing Value Handler**: Automated missing data treatment

---

## 🔬 Testing Suite

The framework includes comprehensive testing to ensure reliability:

- ✅ **Modular Block Testing**: Each component tested independently
- ✅ **Integration Testing**: Complete pipeline workflows
- ✅ **Stress Testing**: Edge cases and error handling  
- ✅ **Local Installation Testing**: Pip extras and local deployment
- ✅ **Dataset Testing**: Auto-download and processing
- ✅ **Model Swapping Testing**: Seamless algorithm switching

*Full testing suite available in separate repository for researchers interested in contributing.*

---

## 🤝 Contributing

We welcome contributions from the HEP community!

### Quick Contribution Guide

1. **New Dataset**: Add to `src/mlpipe/blocks/ingest/` + config in `configs/data/`
2. **New Model**: Add to `src/mlpipe/blocks/model/` + config in `configs/model/`  
3. **New Feature**: Follow existing patterns in relevant `blocks/` subdirectory
4. **Bug Fix**: Tests are in `tests/` directory

### Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/hep-ml-templates.git
cd hep-ml-templates
pip install -e '.[dev]'
pytest tests/
```

---

## 📋 Project Structure

```
hep-ml-templates/
├── src/mlpipe/
│   ├── blocks/              # 🧩 Modular components
│   │   ├── ingest/         #   📊 Data loading (CSV, ROOT, HEP datasets)
│   │   ├── preprocessing/  #   🔧 Data preprocessing & scaling  
│   │   ├── feature_eng/    #   ✨ Feature engineering & selection
│   │   ├── model/          #   🤖 ML models (XGBoost, NN, GNN)
│   │   ├── training/       #   📈 Training procedures
│   │   └── evaluation/     #   📊 Metrics & evaluation
│   ├── core/               # ⚙️  Framework internals
│   ├── pipelines/          # 🚀 Complete pipeline implementations  
│   ├── cli/                # 💻 Command-line interface
│   └── templates/          # 📝 Pipeline templates
├── configs/                # ⚙️  Configuration files
│   ├── pipeline.yaml       #   🎯 Main pipeline orchestrator
│   ├── data/              #   📊 Dataset configurations
│   ├── model/             #   🤖 Model configurations  
│   ├── preprocessing/     #   🔧 Preprocessing configurations
│   └── ...                #   ✨ Other component configs
├── examples/               # 📚 Example implementations
├── data/                   # 📁 Dataset storage (auto-created)
├── docs/                   # 📖 Documentation
└── tests/                  # 🧪 Test suite
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🆘 Getting Help

- **Quick Questions**: Check existing configurations with `mlpipe list-configs`
- **Examples**: See `examples/` directory for complete implementations  
- **Issues**: Open a GitHub issue for bug reports or feature requests
- **Discussions**: Use GitHub Discussions for usage questions

---

**Ready to accelerate your HEP ML research?** 

```bash
pip install 'hep-ml-templates[pipeline-xgb]'
mlpipe run
```

*From raw physics data to trained models in under 30 seconds.* ⚡
- **Change model**: `--overrides model=xgb_classifier` 
- **Change preprocessing**: `--overrides preprocessing=standard`
- **Mix multiple**: `--overrides data=csv_demo model=xgb_classifier`

#### 🎛️ **Pipeline Configuration** (for persistent changes)
- **Edit once, use everywhere**: Modify `configs/pipeline.yaml` to set new defaults
- **Component-specific configs**: Add datasets in `configs/data/`, models in `configs/model/`, etc.
- **Single source of truth**: `pipeline.yaml` orchestrates all components by referencing their configs
- **Share configurations**: Easy to version control and share complete pipeline setups

No need to create new YAML files - just override the components you want to change!

## Tutorials

### Tutorial 1: Quick Component Swapping (Using Overrides)

The fastest way to experiment with different components:

```bash
# Start with default HIGGS dataset
mlpipe run

# Switch to demo dataset with matching features  
mlpipe run --overrides data=csv_demo feature_eng=demo_features

# Try different combinations
mlpipe run --overrides data=csv_demo model=xgb_classifier preprocessing=standard

# See what's available
mlpipe list-configs
```

**When to use**: Quick experiments, testing different combinations, one-off runs.

### Tutorial 2: Persistent Configuration Changes (Editing configs)

For permanent changes or new default setups, edit the main pipeline orchestrator:

#### Step 1: Check available components
```bash
mlpipe list-configs
```

#### Step 2: Edit the pipeline orchestrator
Open `configs/pipeline.yaml`:
```yaml
data: higgs_uci              # ← Change this to csv_demo
preprocessing: standard
feature_eng: column_selector # ← Change this to demo_features  
model: xgb_classifier
training: sklearn
evaluation: classification
runtime: local_cpu
```

#### Step 3: Run with new defaults
```bash
mlpipe run  # Now uses your chosen components as defaults
```

**When to use**: Setting new defaults, sharing configurations, production setups.

### Tutorial 3: Adding a New Dataset

Want to add your own dataset? Follow this pattern:

#### Step 1: Create dataset configuration
Create `configs/data/my_dataset.yaml`:
```yaml
block: ingest.csv
path: data/my_data.csv
label: target
has_header: true
names: null  # or specify column names
label_is_first_column: false
nrows: null  # or limit rows
```

#### Step 2: Update pipeline to use it
Edit `configs/pipeline.yaml`:
```yaml
data: my_dataset  # ← Points to your new configs/data/my_dataset.yaml
preprocessing: standard
feature_eng: column_selector
model: xgb_classifier
training: sklearn
evaluation: classification
runtime: local_cpu
```

#### Step 3: Run your pipeline
```bash
mlpipe run  # Uses your new dataset
```

#### Step 4 (Optional): Create matching feature engineering
If your dataset has different columns, create `configs/feature_eng/my_features.yaml`:
```yaml
block: feature.column_selector
include: [col1, col2, col3]  # Your column names
exclude: []
```

Then update `configs/pipeline.yaml`:
```yaml
data: my_dataset
feature_eng: my_features  # ← Points to your custom features
# ... rest stays the same
```

## Adding a New Component to the Pipeline

To extend the pipeline with a new block (e.g. dataset, feature engineering step, training routine, or model), follow this general process:

1. **Check if it already exists**

   ```bash
   mlpipe list-configs
   ```

   Look under the relevant section (e.g. `model`, `dataset`, `feature_eng`, `training`).

2. **If not present, implement the component**

   * Add the code to the correct folder under `src/mlpipe/blocks/<component>/`.
     For example, a new model goes in:

     ```
     src/mlpipe/blocks/model/<your_model>.py
     ```

3. **Create a configuration file**

   * Add a YAML file describing your component to:

     ```
     configs/<component>/<your_component>.yaml
     ```

4. **Modify the pipeline definition**

   * Open `pipeline.yaml`.
   * Update the relevant section (e.g. `model:`) to point to your new component.
   * Ensure the **name** you use here is consistent with your code and config file.

---

💡 **Note**: The same workflow applies for switching out any block — whether `dataset`, `feature_eng`, `training`, or `model`. Just stay consistent with naming across **code**, **configs**, and **pipeline.yaml**.

---

**Key principle**: `pipeline.yaml` is your single control panel. It orchestrates everything by pointing to the right component configs in `configs/data/`, `configs/model/`, `configs/feature_eng/`, etc.

## Configuration Architecture

The framework uses a **hierarchical configuration system** designed for maximum modularity:

### 📋 Pipeline Level (`configs/pipeline.yaml`)
The **single source of truth** that orchestrates your entire pipeline:
```yaml
data: higgs_uci              # → loads configs/data/higgs_uci.yaml
preprocessing: standard      # → loads configs/preprocessing/standard.yaml  
feature_eng: column_selector # → loads configs/feature_eng/column_selector.yaml
model: xgb_classifier        # → loads configs/model/xgb_classifier.yaml
training: sklearn            # → loads configs/training/sklearn.yaml
evaluation: classification   # → loads configs/evaluation/classification.yaml
```

### 🧩 Component Level (`configs/*/`)
Each component has its own configuration space:
- `configs/data/` - Dataset configurations (paths, formats, preprocessing)
- `configs/model/` - Model architectures and hyperparameters  
- `configs/preprocessing/` - Data preprocessing steps
- `configs/feature_eng/` - Feature engineering configurations
- `configs/training/` - Training procedures and parameters
- `configs/evaluation/` - Evaluation metrics and methods

### 🔄 Two Ways to Change Components
1. **Temporary** (overrides): `mlpipe run --overrides data=csv_demo`
2. **Persistent** (edit pipeline.yaml): Change `data: higgs_uci` → `data: csv_demo`

This design ensures **clean separation of concerns** - each component configuration focuses on its specific domain, while the pipeline configuration handles orchestration.

## Project Structure

```
hep-ml-templates/
├── src/mlpipe/
│   ├── blocks/           # Modular pipeline components
│   │   ├── ingest/       # Data ingestion blocks
│   │   ├── preprocess/   # Preprocessing blocks
│   │   ├── feature_eng/  # Feature engineering blocks
│   │   ├── model/        # ML model blocks
│   │   ├── training/     # Training blocks
│   │   └── evaluation/   # Evaluation blocks
│   ├── core/             # Core framework components
│   ├── pipelines/        # Complete pipeline implementations
│   ├── cli/              # Command-line interface
│   └── templates/        # Pipeline templates
├── configs/              # YAML configuration files
│   ├── data/            # Dataset configurations
│   ├── model/           # Model configurations
│   ├── preprocessing/   # Preprocessing configurations
│   ├── feature_eng/     # Feature engineering configurations
│   ├── training/        # Training configurations
│   └── evaluation/      # Evaluation configurations
├── examples/            # Example pipeline implementations
│   ├── xgb_basic/       # XGBoost classification example
│   ├── ae_basic/        # Autoencoder example
│   └── gnn_basic/       # Graph Neural Network example
├── data/                # Dataset storage
├── docs/                # Documentation
└── tests/               # Unit and integration tests
```

## Supported Datasets

- **HIGGS UCI Dataset**: Particle physics classification task
- Extensible to other HEP datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license here]
