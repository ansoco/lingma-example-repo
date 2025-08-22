# PROJECT_STRUCTURE

## Paper Summary

This paper demonstrates that benign overfitting—where overparameterized models achieve zero training error yet maintain good generalization—occurs in adversarially robust linear classification. The authors prove that gradient descent adversarial training on linearly separable, noisy data can achieve near-optimal standard and adversarial risk despite perfect training fit.

## Repository Tree

```
benign-overfitting-adversarial/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linear_classifier.py      # LinearAdversarialClassifier class
│   │   ├── adversarial_trainer.py    # GradientDescentAdversarialTrainer
│   │   └── loss_functions.py         # ExponentialLoss, adversarial loss computation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic_generator.py    # SubGaussianMixtureGenerator class
│   │   ├── data_loader.py           # DataLoader for train/test splits
│   │   └── noise_injector.py        # LabelNoiseInjector class
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── risk_vs_dimension.py     # Experiment 1: Risk vs dimension d
│   │   ├── risk_vs_iterations.py    # Experiment 2: Risk vs training iterations
│   │   ├── experiment_runner.py     # ExperimentRunner orchestration class
│   │   └── metrics.py               # StandardRisk, AdversarialRisk calculators
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── math_utils.py            # Dual norm calculations, subgradients
│   │   ├── visualization.py         # Plotting functions for results
│   │   └── logging_config.py        # Structured logging setup
│   └── config/
│       ├── __init__.py
│       └── config_manager.py        # Configuration validation and loading
├── tests/
│   ├── __init__.py
│   ├── test_models/
│   │   ├── test_linear_classifier.py
│   │   ├── test_adversarial_trainer.py
│   │   └── test_loss_functions.py
│   ├── test_data/
│   │   ├── test_synthetic_generator.py
│   │   ├── test_data_loader.py
│   │   └── test_noise_injector.py
│   ├── test_experiments/
│   │   ├── test_risk_vs_dimension.py
│   │   ├── test_risk_vs_iterations.py
│   │   └── test_metrics.py
│   └── test_utils/
│       ├── test_math_utils.py
│       └── test_visualization.py
├── configs/
│   ├── base_config.yaml             # Base experimental configuration
│   ├── dimension_experiment.yaml    # Config for dimension experiments
│   ├── iteration_experiment.yaml    # Config for iteration experiments
│   └── model_configs.yaml          # Model hyperparameters
├── results/
│   ├── figures/                     # Generated plots and visualizations
│   ├── logs/                        # Experiment logs and metrics
│   └── checkpoints/                 # Model checkpoints (if needed)
├── requirements.txt                 # Exact dependency versions
├── setup.py                        # Package installation script
├── run_experiments.py              # Main entry point for all experiments
└── README.md                       # Usage instructions and reproduction guide
```

## Module Interface Map

### Core Model Classes

```python
# src/models/linear_classifier.py
class LinearAdversarialClassifier:
    def __init__(self, input_dim: int, initialization: str = "xavier_normal")
    def forward(self, x: np.ndarray) -> np.ndarray  # Returns sign(θᵀx)
    def get_parameters(self) -> np.ndarray  # Returns θ
    def set_parameters(self, theta: np.ndarray) -> None
    def predict(self, x: np.ndarray) -> np.ndarray  # Binary predictions {-1, +1}

# src/models/adversarial_trainer.py
class GradientDescentAdversarialTrainer:
    def __init__(self, model: LinearAdversarialClassifier, 
                 learning_rate: float, p_norm: float, epsilon: float)
    def compute_adversarial_loss(self, x: np.ndarray, y: np.ndarray) -> float
    def compute_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float  # Returns loss
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              num_iterations: int) -> Dict[str, List[float]]  # Returns training history

# src/models/loss_functions.py
class ExponentialLoss:
    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float
    @staticmethod
    def compute_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray

class AdversarialExponentialLoss:
    def __init__(self, p_norm: float, epsilon: float)
    def compute_loss(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float
    def compute_gradient(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray
```

### Data Generation and Processing

```python
# src/data/synthetic_generator.py
class SubGaussianMixtureGenerator:
    def __init__(self, dimension: int, mu_norm: float, noise_type: str = "gaussian")
    def generate_clean_data(self, n_samples: int, random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]
    def get_mu_vector(self) -> np.ndarray
    def get_separation_margin(self) -> float

# src/data/noise_injector.py
class LabelNoiseInjector:
    def __init__(self, noise_rate: float)
    def inject_noise(self, y_clean: np.ndarray, random_seed: int = None) -> np.ndarray
    def get_noise_mask(self, y_clean: np.ndarray) -> np.ndarray

# src/data/data_loader.py
class DataLoader:
    def __init__(self, generator: SubGaussianMixtureGenerator, 
                 noise_injector: LabelNoiseInjector)
    def generate_dataset(self, n_train: int, n_test: int, 
                        random_seed: int = None) -> Dict[str, np.ndarray]
    def split_data(self, x: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.8) -> Dict[str, np.ndarray]
```

### Experiment Framework

```python
# src/experiments/experiment_runner.py
class ExperimentRunner:
    def __init__(self, config: Dict[str, Any])
    def run_dimension_experiment(self) -> Dict[str, Any]
    def run_iteration_experiment(self) -> Dict[str, Any]
    def save_results(self, results: Dict[str, Any], experiment_name: str) -> None
    def load_results(self, experiment_name: str) -> Dict[str, Any]

# src/experiments/metrics.py
class RiskCalculator:
    @staticmethod
    def compute_standard_risk(model: LinearAdversarialClassifier, 
                            x_test: np.ndarray, y_test: np.ndarray) -> float
    @staticmethod
    def compute_adversarial_risk(model: LinearAdversarialClassifier,
                               x_test: np.ndarray, y_test: np.ndarray,
                               p_norm: float, epsilon: float) -> float
    @staticmethod
    def compute_training_error(model: LinearAdversarialClassifier,
                             x_train: np.ndarray, y_train: np.ndarray) -> float
```

## Technology Stack

- **Programming Language**: Python 3.9+
- **Core Libraries**:
  - `numpy==1.24.3`: Numerical computations, linear algebra operations
  - `scipy==1.10.1`: Statistical functions, optimization utilities
  - `matplotlib==3.7.1`: Plotting and visualization
  - `seaborn==0.12.2`: Statistical plotting enhancements
  - `pyyaml==6.0`: Configuration file parsing
  - `tqdm==4.65.0`: Progress bars for long-running experiments
  - `pytest==7.3.1`: Unit testing framework
  - `pytest-cov==4.1.0`: Test coverage reporting
- **Hardware Requirements**:
  - CPU: Multi-core processor (experiments are CPU-intensive)
  - RAM: 8GB minimum (for high-dimensional experiments with d=2500)
  - Storage: 1GB for results and logs
  - GPU: Not required (linear models, small datasets)

## Configuration Schema

```yaml
# Base configuration structure
experiment:
  name: str                    # Experiment identifier
  random_seed: int            # Global random seed for reproducibility
  output_dir: str             # Results output directory

data:
  n_train: int               # Training samples (default: 50)
  n_test: int                # Test samples (default: 2000)
  noise_rate: float          # Label noise η (default: 0.1)
  noise_type: str            # "gaussian" or "subgaussian"

model:
  initialization: str        # "xavier_normal" or "zeros"
  
training:
  learning_rate: float       # Gradient descent step size α
  num_iterations: int        # Training iterations T
  p_norm: float             # Adversarial perturbation norm (2.0 or inf)
  epsilon: float            # Perturbation budget ε

dimension_experiment:
  dimensions: List[int]      # List of dimensions to test
  mu_scalings: List[float]   # μ scaling factors (e.g., [0.2, 0.3, 0.4])
  
iteration_experiment:
  fixed_dimension: int       # Fixed d for iteration experiment
  mu_scaling: float         # Fixed μ scaling
  epsilon_values: List[float] # Different ε values to test
```

## Entry Points

### Command Line Interface

```bash
# Main experiment runner
python run_experiments.py --config configs/dimension_experiment.yaml --experiment dimension
python run_experiments.py --config configs/iteration_experiment.yaml --experiment iteration

# Individual experiment modules
python -m src.experiments.risk_vs_dimension --config configs/dimension_experiment.yaml
python -m src.experiments.risk_vs_iterations --config configs/iteration_experiment.yaml

# Testing
pytest tests/ --cov=src/ --cov-report=html
python -m pytest tests/test_models/ -v
```

### Python API Entry Points

```python
# Primary execution function
def main(config_path: str, experiment_type: str) -> None:
    """Main entry point for experiment execution"""

# Individual experiment functions  
def run_dimension_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute risk vs dimension experiment"""

def run_iteration_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute risk vs iteration experiment"""

# Utility functions
def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging"""

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
```

---