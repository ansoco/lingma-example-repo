# IMPLEMENTATION_PLAN

## Implementation Overview

This implementation plan provides a sequential roadmap for reproducing the "Benign Overfitting in Adversarially Robust Linear Classification" paper. The workflow progresses from basic infrastructure setup through data pipeline creation, model implementation, and finally experimental validation. Each phase builds upon previous components with clear validation checkpoints.

**Expected Timeline**: 4-6 hours for complete implementation
**Success Criteria**: Reproduce Figure 1 results showing benign overfitting phenomenon

## Phase-Based Development

### Phase 1: Core Infrastructure (30 minutes)

#### Environment Setup and Dependencies

**CLAUDE CODE TASK**: Create virtual environment and install dependencies

```bash
# Create project structure
mkdir -p benign-overfitting-adversarial/{src/{models,data,experiments,utils,config},tests/{test_models,test_data,test_experiments,test_utils},configs,results/{figures,logs,checkpoints}}

# Create requirements.txt with exact versions
cat > requirements.txt << EOF
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.1
seaborn==0.12.2
pyyaml==6.0
tqdm==4.65.0
pytest==7.3.1
pytest-cov==4.1.0
EOF
```

#### Configuration System Implementation

**File**: `src/config/config_manager.py`

```python
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self) -> None:
        """Validate required configuration parameters"""
        required_sections = ['experiment', 'data', 'model', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate specific parameters
        assert self.config['data']['n_train'] > 0
        assert 0 <= self.config['data']['noise_rate'] <= 1
        assert self.config['training']['learning_rate'] > 0
        assert self.config['training']['epsilon'] >= 0
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, default)
            if value is None:
                return default
        return value
```

#### Logging and Monitoring Setup

**File**: `src/utils/logging_config.py`

```python
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure structured logging for experiments"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger('benign_overfitting')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
```

**CLAUDE CODE TASK**: Create base configuration files in `configs/` directory

```yaml
# configs/base_config.yaml
experiment:
  name: "benign_overfitting_reproduction"
  random_seed: 42
  output_dir: "results"

data:
  n_train: 50
  n_test: 2000
  noise_rate: 0.1
  noise_type: "gaussian"

model:
  initialization: "xavier_normal"

training:
  learning_rate: 0.001
  num_iterations: 1000
  p_norm: 2.0
  epsilon: 0.1
```

**Validation Checkpoint**: Run configuration loading test
```python
# Test configuration system
config = ConfigManager('configs/base_config.yaml')
assert config.get('data.n_train') == 50
assert config.get('training.learning_rate') == 0.001
print("✅ Phase 1 Complete: Infrastructure setup successful")
```

### Phase 2: Data Pipeline (45 minutes)

#### Mathematical Utilities Implementation

**File**: `src/utils/math_utils.py`

```python
import numpy as np
from typing import Tuple, Union

class DualNormCalculator:
    """Utilities for computing dual norms and subgradients"""
    
    @staticmethod
    def compute_dual_norm(theta: np.ndarray, p: float) -> float:
        """Compute ||θ||_q where 1/p + 1/q = 1"""
        if p == 1:
            return np.max(np.abs(theta))  # L∞ norm
        elif p == 2:
            return np.linalg.norm(theta, ord=2)  # L2 norm
        elif np.isinf(p):
            return np.linalg.norm(theta, ord=1)  # L1 norm
        else:
            q = p / (p - 1)  # Dual exponent
            return np.linalg.norm(theta, ord=q)
    
    @staticmethod
    def compute_dual_norm_subgradient(theta: np.ndarray, p: float) -> np.ndarray:
        """Compute subgradient of ||θ||_q"""
        if p == 1:
            # Subgradient of L∞ norm
            max_indices = np.abs(theta) == np.max(np.abs(theta))
            subgrad = np.zeros_like(theta)
            subgrad[max_indices] = np.sign(theta[max_indices])
            return subgrad / np.sum(max_indices)
        elif p == 2:
            # Gradient of L2 norm
            norm = np.linalg.norm(theta, ord=2)
            return theta / (norm + 1e-8)  # Add small epsilon for numerical stability
        elif np.isinf(p):
            # Subgradient of L1 norm
            return np.sign(theta)
        else:
            q = p / (p - 1)
            norm_q = np.linalg.norm(theta, ord=q)
            if norm_q == 0:
                return np.zeros_like(theta)
            return (np.abs(theta) ** (q-1) * np.sign(theta)) / (norm_q ** (q-1))

def generate_subgaussian_noise(shape: Tuple[int, ...], 
                              subgaussian_norm: float = 1.0,
                              random_state: np.random.RandomState = None) -> np.ndarray:
    """Generate sub-Gaussian noise with specified norm"""
    if random_state is None:
        random_state = np.random.RandomState()
    
    # For simplicity, use Gaussian noise (which is sub-Gaussian)
    return random_state.normal(0, subgaussian_norm, shape)
```

#### Synthetic Data Generation

**File**: `src/data/synthetic_generator.py`

```python
import numpy as np
from typing import Tuple, Optional
from ..utils.math_utils import generate_subgaussian_noise

class SubGaussianMixtureGenerator:
    """Generate synthetic data following the paper's sub-Gaussian mixture model"""
    
    def __init__(self, dimension: int, mu_scaling: float, noise_type: str = "gaussian"):
        """
        Args:
            dimension: Feature dimension d
            mu_scaling: Scaling factor for ||μ||₂ = d^mu_scaling
            noise_type: Type of noise distribution
        """
        self.dimension = dimension
        self.mu_scaling = mu_scaling
        self.noise_type = noise_type
        
        # Generate mean vector μ
        # Set μ = (d^mu_scaling / √d) * e₁ where e₁ is first standard basis vector
        self.mu = np.zeros(dimension)
        if dimension > 0:
            mu_norm = dimension ** mu_scaling
            self.mu[0] = mu_norm / np.sqrt(dimension)
    
    def generate_clean_data(self, n_samples: int, 
                          random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate clean data (x̃, ỹ) following: x̃ = ỹμ + ξ
        
        Returns:
            x_clean: (n_samples, dimension) feature matrix
            y_clean: (n_samples,) label vector in {-1, +1}
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random_state = np.random.RandomState(random_seed)
        else:
            random_state = np.random.RandomState()
        
        # Generate labels ỹ uniformly from {-1, +1}
        y_clean = random_state.choice([-1, 1], size=n_samples)
        
        # Generate noise ξ ~ sub-Gaussian with norm ≤ 1
        if self.noise_type == "gaussian":
            xi = random_state.normal(0, 1, size=(n_samples, self.dimension))
        else:
            xi = generate_subgaussian_noise((n_samples, self.dimension), 
                                          subgaussian_norm=1.0, 
                                          random_state=random_state)
        
        # Generate features: x̃ = ỹμ + ξ
        x_clean = y_clean.reshape(-1, 1) * self.mu.reshape(1, -1) + xi
        
        return x_clean, y_clean
    
    def get_mu_vector(self) -> np.ndarray:
        """Return the mean separation vector μ"""
        return self.mu.copy()
    
    def get_separation_margin(self) -> float:
        """Return the margin γ̄ = ||μ||₂"""
        return np.linalg.norm(self.mu, ord=2)
```

#### Label Noise Injection

**File**: `src/data/noise_injector.py`

```python
import numpy as np
from typing import Optional

class LabelNoiseInjector:
    """Inject label noise to simulate noisy training distribution"""
    
    def __init__(self, noise_rate: float):
        """
        Args:
            noise_rate: Probability η of label flip (0 ≤ η ≤ 1)
        """
        assert 0 <= noise_rate <= 1, "Noise rate must be between 0 and 1"
        self.noise_rate = noise_rate
    
    def inject_noise(self, y_clean: np.ndarray, 
                    random_seed: Optional[int] = None) -> np.ndarray:
        """
        Inject label noise: flip each label with probability η
        
        Args:
            y_clean: Clean labels in {-1, +1}
            random_seed: Random seed for reproducibility
            
        Returns:
            y_noisy: Noisy labels in {-1, +1}
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        y_noisy = y_clean.copy()
        n_samples = len(y_clean)
        
        # Generate flip mask
        flip_mask = np.random.random(n_samples) < self.noise_rate
        
        # Flip labels
        y_noisy[flip_mask] = -y_noisy[flip_mask]
        
        return y_noisy
    
    def get_noise_mask(self, y_clean: np.ndarray, y_noisy: np.ndarray) -> np.ndarray:
        """Return boolean mask indicating which labels were flipped"""
        return y_clean != y_noisy
```

#### Data Loading and Management

**File**: `src/data/data_loader.py`

```python
import numpy as np
from typing import Dict, Optional, Tuple
from .synthetic_generator import SubGaussianMixtureGenerator
from .noise_injector import LabelNoiseInjector

class DataLoader:
    """Orchestrate data generation, noise injection, and train/test splitting"""
    
    def __init__(self, generator: SubGaussianMixtureGenerator, 
                 noise_injector: LabelNoiseInjector):
        self.generator = generator
        self.noise_injector = noise_injector
    
    def generate_dataset(self, n_train: int, n_test: int, 
                        random_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate complete dataset with train/test split
        
        Returns:
            Dictionary with keys: 'x_train', 'y_train', 'x_test', 'y_test',
                                'y_train_clean', 'y_test_clean'
        """
        if random_seed is not None:
            # Use different seeds for train and test to avoid overlap
            train_seed = random_seed
            test_seed = random_seed + 1000
        else:
            train_seed = test_seed = None
        
        # Generate training data
        x_train, y_train_clean = self.generator.generate_clean_data(
            n_train, random_seed=train_seed
        )
        y_train = self.noise_injector.inject_noise(
            y_train_clean, random_seed=train_seed
        )
        
        # Generate test data (clean labels for evaluation)
        x_test, y_test_clean = self.generator.generate_clean_data(
            n_test, random_seed=test_seed
        )
        # Test data uses clean labels (no noise injection)
        y_test = y_test_clean.copy()
        
        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'y_train_clean': y_train_clean,
            'y_test_clean': y_test_clean
        }
    
    @staticmethod
    def compute_data_statistics(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute and return dataset statistics"""
        stats = {
            'n_train': len(data['y_train']),
            'n_test': len(data['y_test']),
            'dimension': data['x_train'].shape[1],
            'train_noise_rate': np.mean(data['y_train'] != data['y_train_clean']),
            'train_balance': np.mean(data['y_train'] == 1),
            'test_balance': np.mean(data['y_test'] == 1),
        }
        return stats
```

**CLAUDE CODE TASK**: Implement data pipeline validation tests

```python
# Test data generation pipeline
def test_data_pipeline():
    generator = SubGaussianMixtureGenerator(dimension=10, mu_scaling=0.3)
    noise_injector = LabelNoiseInjector(noise_rate=0.1)
    loader = DataLoader(generator, noise_injector)
    
    data = loader.generate_dataset(n_train=50, n_test=100, random_seed=42)
    stats = loader.compute_data_statistics(data)
    
    # Validate shapes and properties
    assert data['x_train'].shape == (50, 10)
    assert data['x_test'].shape == (100, 10)
    assert np.all(np.isin(data['y_train'], [-1, 1]))
    assert 0.05 <= stats['train_noise_rate'] <= 0.15  # Approximately 10% noise
    
    print("✅ Phase 2 Complete: Data pipeline validated")

test_data_pipeline()
```

### Phase 3: Model Implementation (60 minutes)

#### Loss Functions Implementation

**File**: `src/models/loss_functions.py`

```python
import numpy as np
from typing import Tuple
from ..utils.math_utils import DualNormCalculator

class ExponentialLoss:
    """Standard exponential loss: exp(-yθᵀx)"""
    
    @staticmethod
    def compute_loss(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute exponential loss over dataset
        
        Args:
            theta: Model parameters (d,)
            x: Features (n, d)
            y: Labels (n,) in {-1, +1}
            
        Returns:
            Average exponential loss
        """
        margins = y * (x @ theta)  # yᵢθᵀxᵢ for each sample
        losses = np.exp(-margins)
        return np.mean(losses)
    
    @staticmethod
    def compute_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of exponential loss
        
        Returns:
            Gradient vector (d,)
        """
        margins = y * (x @ theta)
        weights = np.exp(-margins)  # exp(-yᵢθᵀxᵢ)
        
        # Gradient: -∑ᵢ yᵢxᵢ exp(-yᵢθᵀxᵢ)
        gradient = -np.mean((y * weights).reshape(-1, 1) * x, axis=0)
        return gradient

class AdversarialExponentialLoss:
    """Adversarial exponential loss with ℓₚ perturbations"""
    
    def __init__(self, p_norm: float, epsilon: float):
        """
        Args:
            p_norm: Norm constraint for adversarial perturbations
            epsilon: Perturbation budget ε
        """
        self.p_norm = p_norm
        self.epsilon = epsilon
        self.dual_calc = DualNormCalculator()
    
    def compute_loss(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute adversarial exponential loss
        
        For linear models: max_{x' ∈ Bₚᵋ(x)} exp(-yθᵀx') = exp(-yθᵀx + ε||θ||_q)
        """
        margins = y * (x @ theta)  # yᵢθᵀxᵢ
        dual_norm = self.dual_calc.compute_dual_norm(theta, self.p_norm)
        
        # Adversarial loss: exp(-yᵢθᵀxᵢ + ε||θ||_q)
        adv_losses = np.exp(-margins + self.epsilon * dual_norm)
        return np.mean(adv_losses)
    
    def compute_gradient(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of adversarial loss
        
        ∇θ L = -∑ᵢ (yᵢxᵢ - ε∂||θ||_q) exp(-yᵢθᵀxᵢ + ε||θ||_q)
        """
        margins = y * (x @ theta)
        dual_norm = self.dual_calc.compute_dual_norm(theta, self.p_norm)
        dual_subgrad = self.dual_calc.compute_dual_norm_subgradient(theta, self.p_norm)
        
        # Weights for each sample
        weights = np.exp(-margins + self.epsilon * dual_norm)
        
        # Gradient components
        data_term = np.mean((y * weights).reshape(-1, 1) * x, axis=0)
        regularization_term = self.epsilon * dual_subgrad * np.mean(weights)
        
        gradient = -data_term + regularization_term
        return gradient
```

#### Linear Classifier Implementation

**File**: `src/models/linear_classifier.py`

```python
import numpy as np
from typing import Optional

class LinearAdversarialClassifier:
    """Linear classifier for adversarial training"""
    
    def __init__(self, input_dim: int, initialization: str = "xavier_normal"):
        """
        Args:
            input_dim: Input feature dimension
            initialization: Parameter initialization method
        """
        self.input_dim = input_dim
        self.initialization = initialization
        self.theta = self._initialize_parameters()
    
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize model parameters"""
        if self.initialization == "zeros":
            return np.zeros(self.input_dim)
        elif self.initialization == "xavier_normal":
            # Xavier normal initialization
            std = np.sqrt(2.0 / self.input_dim)
            return np.random.normal(0, std, self.input_dim)
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute θᵀx
        
        Args:
            x: Input features (n, d) or (d,)
            
        Returns:
            Linear outputs (n,) or scalar
        """
        return x @ self.theta
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make binary predictions: sign(θᵀx)
        
        Returns:
            Predictions in {-1, +1}
        """
        outputs = self.forward(x)
        return np.sign(outputs)
    
    def get_parameters(self) -> np.ndarray:
        """Return copy of current parameters"""
        return self.theta.copy()
    
    def set_parameters(self, theta: np.ndarray) -> None:
        """Set model parameters"""
        assert len(theta) == self.input_dim
        self.theta = theta.copy()
    
    def get_parameter_norm(self, p: float = 2.0) -> float:
        """Compute ||θ||_p"""
        if p == 2:
            return np.linalg.norm(self.theta, ord=2)
        elif p == 1:
            return np.linalg.norm(self.theta, ord=1)
        elif np.isinf(p):
            return np.linalg.norm(self.theta, ord=np.inf)
        else:
            return np.linalg.norm(self.theta, ord=p)
```

#### Adversarial Training Implementation

**File**: `src/models/adversarial_trainer.py`

```python
import numpy as np
from typing import Dict, List, Optional
from .linear_classifier import LinearAdversarialClassifier
from .loss_functions import AdversarialExponentialLoss
import logging

class GradientDescentAdversarialTrainer:
    """Gradient descent adversarial training for linear classifiers"""
    
    def __init__(self, model: LinearAdversarialClassifier, 
                 learning_rate: float, p_norm: float, epsilon: float):
        """
        Args:
            model: Linear classifier to train
            learning_rate: Gradient descent step size α
            p_norm: Adversarial perturbation norm
            epsilon: Perturbation budget ε
        """
        self.model = model
        self.learning_rate = learning_rate
        self.p_norm = p_norm
        self.epsilon = epsilon
        self.loss_fn = AdversarialExponentialLoss(p_norm, epsilon)
        self.logger = logging.getLogger('benign_overfitting.trainer')
    
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Perform single gradient descent step
        
        Returns:
            Current loss value
        """
        # Compute current loss
        current_loss = self.loss_fn.compute_loss(self.model.theta, x, y)
        
        # Compute gradient
        gradient = self.loss_fn.compute_gradient(self.model.theta, x, y)
        
        # Update parameters: θ ← θ - α∇L(θ)
        new_theta = self.model.theta - self.learning_rate * gradient
        self.model.set_parameters(new_theta)
        
        return current_loss
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              num_iterations: int, log_interval: int = 100) -> Dict[str, List[float]]:
        """
        Train model using gradient descent adversarial training
        
        Args:
            x_train: Training features (n, d)
            y_train: Training labels (n,) in {-1, +1}
            num_iterations: Number of training iterations T
            log_interval: Logging frequency
            
        Returns:
            Training history dictionary
        """
        history = {
            'losses': [],
            'parameter_norms': [],
            'training_errors': []
        }
        
        self.logger.info(f"Starting adversarial training for {num_iterations} iterations")
        self.logger.info(f"p_norm={self.p_norm}, epsilon={self.epsilon}, lr={self.learning_rate}")
        
        for iteration in range(num_iterations):
            # Perform training step
            loss = self.train_step(x_train, y_train)
            
            # Record metrics
            history['losses'].append(loss)
            history['parameter_norms'].append(self.model.get_parameter_norm(p=2))
            
            # Compute training error
            predictions = self.model.predict(x_train)
            training_error = np.mean(predictions != y_train)
            history['training_errors'].append(training_error)
            
            # Logging
            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                self.logger.info(
                    f"Iter {iteration:4d}: Loss={loss:.6f}, "
                    f"TrainErr={training_error:.4f}, ||θ||₂={history['parameter_norms'][-1]:.4f}"
                )
        
        self.logger.info("Training completed")
        return history
    
    def compute_training_error(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute classification error on given data"""
        predictions = self.model.predict(x)
        return np.mean(predictions != y)
```

**CLAUDE CODE TASK**: Implement model validation tests

```python
# Test model implementation
def test_model_implementation():
    # Test linear classifier
    model = LinearAdversarialClassifier(input_dim=5, initialization="zeros")
    x_test = np.random.randn(10, 5)
    
    # Test forward pass
    outputs = model.forward(x_test)
    assert outputs.shape == (10,)
    
    # Test predictions
    preds = model.predict(x_test)
    assert np.all(np.isin(preds, [-1, 1]))
    
    # Test adversarial trainer
    y_test = np.random.choice([-1, 1], size=10)
    trainer = GradientDescentAdversarialTrainer(
        model=model, learning_rate=0.01, p_norm=2.0, epsilon=0.1
    )
    
    # Test single training step
    initial_loss = trainer.loss_fn.compute_loss(model.theta, x_test, y_test)
    step_loss = trainer.train_step(x_test, y_test)
    
    print(f"Initial loss: {initial_loss:.4f}, Step loss: {step_loss:.4f}")
    print("✅ Phase 3 Complete: Model implementation validated")

test_model_implementation()
```

### Phase 4: Experiments and Evaluation (90 minutes)

#### Risk Calculation Metrics

**File**: `src/experiments/metrics.py`

```python
import numpy as np
from typing import Tuple
from ..models.linear_classifier import LinearAdversarialClassifier
from ..utils.math_utils import DualNormCalculator

class RiskCalculator:
    """Calculate standard and adversarial risk metrics"""
    
    @staticmethod
    def compute_standard_risk(model: LinearAdversarialClassifier, 
                            x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Compute standard risk: P[f(x) ≠ y] on clean test data
        
        Returns:
            Classification error rate
        """
        predictions = model.predict(x_test)
        return np.mean(predictions != y_test)
    
    @staticmethod
    def compute_adversarial_risk(model: LinearAdversarialClassifier,
                               x_test: np.ndarray, y_test: np.ndarray,
                               p_norm: float, epsilon: float) -> float:
        """
        Compute adversarial risk: P[∃x' ∈ Bₚᵋ(x) s.t. f(x') ≠ y]
        
        For linear classifiers, this has a closed form solution.
        """
        theta = model.get_parameters()
        dual_calc = DualNormCalculator()
        
        # Compute margins: yᵢθᵀxᵢ
        margins = y_test * (x_test @ theta)
        
        # Compute dual norm ||θ||_q
        dual_norm = dual_calc.compute_dual_norm(theta, p_norm)
        
        # Adversarial margin: yᵢθᵀxᵢ - ε||θ||_q
        # Adversarial error occurs when adversarial margin ≤ 0
        adversarial_margins = margins - epsilon * dual_norm
        
        # Count samples with non-positive adversarial margin
        adversarial_errors = adversarial_margins <= 0
        return np.mean(adversarial_errors)
    
    @staticmethod
    def compute_training_error(model: LinearAdversarialClassifier,
                             x_train: np.ndarray, y_train: np.ndarray) -> float:
        """Compute training classification error"""
        predictions = model.predict(x_train)
        return np.mean(predictions != y_train)
    
    @staticmethod
    def compute_all_risks(model: LinearAdversarialClassifier,
                         x_train: np.ndarray, y_train: np.ndarray,
                         x_test: np.ndarray, y_test: np.ndarray,
                         p_norm: float, epsilon: float) -> Dict[str, float]:
        """Compute all risk metrics"""
        return {
            'training_error': RiskCalculator.compute_training_error(model, x_train, y_train),
            'standard_risk': RiskCalculator.compute_standard_risk(model, x_test, y_test),
            'adversarial_risk': RiskCalculator.compute_adversarial_risk(
                model, x_test, y_test, p_norm, epsilon
            )
        }
```

#### Risk vs Dimension Experiment

**File**: `src/experiments/risk_vs_dimension.py`

```python
import numpy as np
from typing import Dict, List, Any
import logging
from tqdm import tqdm

from ..data.synthetic_generator import SubGaussianMixtureGenerator
from ..data.noise_injector import LabelNoiseInjector
from ..data.data_loader import DataLoader
from ..models.linear_classifier import LinearAdversarialClassifier
from ..models.adversarial_trainer import GradientDescentAdversarialTrainer
from .metrics import RiskCalculator

class RiskVsDimensionExperiment:
    """Experiment 1: Study risk vs dimension d (Figure 1 a-d)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('benign_overfitting.experiment')
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run risk vs dimension experiment
        
        Returns:
            Results dictionary with risks for each dimension and μ scaling
        """
        dimensions = self.config['dimension_experiment']['dimensions']
        mu_scalings = self.config['dimension_experiment']['mu_scalings']
        
        results = {
            'dimensions': dimensions,
            'mu_scalings': mu_scalings,
            'results': {}
        }
        
        # Extract experiment parameters
        n_train = self.config['data']['n_train']
        n_test = self.config['data']['n_test']
        noise_rate = self.config['data']['noise_rate']
        learning_rate = self.config['training']['learning_rate']
        num_iterations = self.config['training']['num_iterations']
        p_norm = self.config['training']['p_norm']
        epsilon = self.config['training']['epsilon']
        random_seed = self.config['experiment']['random_seed']
        
        self.logger.info(f"Starting risk vs dimension experiment")
        self.logger.info(f"Dimensions: {dimensions}")
        self.logger.info(f"μ scalings: {mu_scalings}")
        
        for mu_scaling in mu_scalings:
            self.logger.info(f"Running experiments for μ scaling = {mu_scaling}")
            
            scaling_results = {
                'standard_risks': [],
                'adversarial_risks': [],
                'training_errors': []
            }
            
            for dimension in tqdm(dimensions, desc=f"μ_scaling={mu_scaling}"):
                # Generate data
                generator = SubGaussianMixtureGenerator(
                    dimension=dimension, 
                    mu_scaling=mu_scaling
                )
                noise_injector = LabelNoiseInjector(noise_rate=noise_rate)
                loader = DataLoader(generator, noise_injector)
                
                data = loader.generate_dataset(
                    n_train=n_train, 
                    n_test=n_test, 
                    random_seed=random_seed
                )
                
                # Train model
                model = LinearAdversarialClassifier(
                    input_dim=dimension, 
                    initialization=self.config['model']['initialization']
                )
                
                trainer = GradientDescentAdversarialTrainer(
                    model=model,
                    learning_rate=learning_rate,
                    p_norm=p_norm,
                    epsilon=epsilon
                )
                
                # Train with minimal logging for speed
                trainer.train(
                    data['x_train'], 
                    data['y_train'], 
                    num_iterations=num_iterations,
                    log_interval=num_iterations  # Only log at end
                )
                
                # Compute risks
                risks = RiskCalculator.compute_all_risks(
                    model=model,
                    x_train=data['x_train'],
                    y_train=data['y_train'],
                    x_test=data['x_test'],
                    y_test=data['y_test'],
                    p_norm=p_norm,
                    epsilon=epsilon
                )
                
                scaling_results['standard_risks'].append(risks['standard_risk'])
                scaling_results['adversarial_risks'].append(risks['adversarial_risk'])
                scaling_results['training_errors'].append(risks['training_error'])
                
                self.logger.debug(
                    f"d={dimension}: StdRisk={risks['standard_risk']:.4f}, "
                    f"AdvRisk={risks['adversarial_risk']:.4f}, "
                    f"TrainErr={risks['training_error']:.4f}"
                )
            
            results['results'][f'mu_scaling_{mu_scaling}'] = scaling_results
        
        self.logger.info("Risk vs dimension experiment completed")
        return results
```

#### Risk vs Iterations Experiment

**File**: `src/experiments/risk_vs_iterations.py`

```python
import numpy as np
from typing import Dict, List, Any
import logging
from tqdm import tqdm

from ..data.synthetic_generator import SubGaussianMixtureGenerator
from ..data.noise_injector import LabelNoiseInjector
from ..data.data_loader import DataLoader
from ..models.linear_classifier import LinearAdversarialClassifier
from ..models.adversarial_trainer import GradientDescentAdversarialTrainer
from .metrics import RiskCalculator

class RiskVsIterationsExperiment:
    """Experiment 2: Study risk vs training iterations (Figure 1 e-f)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('benign_overfitting.experiment')
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run risk vs iterations experiment
        
        Returns:
            Results dictionary with risks over training iterations
        """
        # Extract parameters
        fixed_dimension = self.config['iteration_experiment']['fixed_dimension']
        mu_scaling = self.config['iteration_experiment']['mu_scaling']
        epsilon_values = self.config['iteration_experiment']['epsilon_values']
        
        n_train = self.config['data']['n_train']
        n_test = self.config['data']['n_test']
        noise_rate = self.config['data']['noise_rate']
        learning_rate = self.config['training']['learning_rate']
        num_iterations = self.config['training']['num_iterations']
        p_norm = self.config['training']['p_norm']
        random_seed = self.config['experiment']['random_seed']
        
        results = {
            'fixed_dimension': fixed_dimension,
            'mu_scaling': mu_scaling,
            'epsilon_values': epsilon_values,
            'num_iterations': num_iterations,
            'results': {}
        }
        
        self.logger.info(f"Starting risk vs iterations experiment")
        self.logger.info(f"Fixed dimension: {fixed_dimension}")
        self.logger.info(f"μ scaling: {mu_scaling}")
        self.logger.info(f"ε values: {epsilon_values}")
        
        # Generate fixed dataset
        generator = SubGaussianMixtureGenerator(
            dimension=fixed_dimension, 
            mu_scaling=mu_scaling
        )
        noise_injector = LabelNoiseInjector(noise_rate=noise_rate)
        loader = DataLoader(generator, noise_injector)
        
        data = loader.generate_dataset(
            n_train=n_train, 
            n_test=n_test, 
            random_seed=random_seed
        )
        
        for epsilon in epsilon_values:
            self.logger.info(f"Running experiment for ε = {epsilon}")
            
            # Initialize model
            model = LinearAdversarialClassifier(
                input_dim=fixed_dimension,
                initialization=self.config['model']['initialization']
            )
            
            trainer = GradientDescentAdversarialTrainer(
                model=model,
                learning_rate=learning_rate,
                p_norm=p_norm,
                epsilon=epsilon
            )
            
            # Track risks over iterations
            iteration_results = {
                'iterations': [],
                'standard_risks': [],
                'adversarial_risks': [],
                'training_errors': [],
                'losses': []
            }
            
            # Evaluate at regular intervals
            eval_interval = max(1, num_iterations // 50)  # 50 evaluation points
            
            for iteration in tqdm(range(num_iterations + 1), desc=f"ε={epsilon}"):
                # Evaluate at start and at intervals
                if iteration % eval_interval == 0 or iteration == num_iterations:
                    risks = RiskCalculator.compute_all_risks(
                        model=model,
                        x_train=data['x_train'],
                        y_train=data['y_train'],
                        x_test=data['x_test'],
                        y_test=data['y_test'],
                        p_norm=p_norm,
                        epsilon=epsilon
                    )
                    
                    current_loss = trainer.loss_fn.compute_loss(
                        model.theta, data['x_train'], data['y_train']
                    )
                    
                    iteration_results['iterations'].append(iteration)
                    iteration_results['standard_risks'].append(risks['standard_risk'])
                    iteration_results['adversarial_risks'].append(risks['adversarial_risk'])
                    iteration_results['training_errors'].append(risks['training_error'])
                    iteration_results['losses'].append(current_loss)
                
                # Perform training step (except at final iteration)
                if iteration < num_iterations:
                    trainer.train_step(data['x_train'], data['y_train'])
            
            results['results'][f'epsilon_{epsilon}'] = iteration_results
        
        self.logger.info("Risk vs iterations experiment completed")
        return results
```

#### Experiment Runner and Orchestration

**File**: `src/experiments/experiment_runner.py`

```python
import json
import pickle
from pathlib import Path
from typing import Dict, Any
import logging

from .risk_vs_dimension import RiskVsDimensionExperiment
from .risk_vs_iterations import RiskVsIterationsExperiment
from ..utils.visualization import ExperimentVisualizer

class ExperimentRunner:
    """Orchestrate and manage all experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('benign_overfitting.runner')
    
    def run_dimension_experiment(self) -> Dict[str, Any]:
        """Run and save dimension experiment"""
        self.logger.info("Starting dimension experiment")
        
        experiment = RiskVsDimensionExperiment(self.config)
        results = experiment.run_experiment()
        
        # Save results
        self.save_results(results, 'dimension_experiment')
        
        # Generate plots
        visualizer = ExperimentVisualizer(self.config)
        visualizer.plot_risk_vs_dimension(results)
        
        return results
    
    def run_iteration_experiment(self) -> Dict[str, Any]:
        """Run and save iteration experiment"""
        self.logger.info("Starting iteration experiment")
        
        experiment = RiskVsIterationsExperiment(self.config)
        results = experiment.run_experiment()
        
        # Save results
        self.save_results(results, 'iteration_experiment')
        
        # Generate plots
        visualizer = ExperimentVisualizer(self.config)
        visualizer.plot_risk_vs_iterations(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any], experiment_name: str) -> None:
        """Save experiment results"""
        # Save as JSON (human readable)
        json_path = self.output_dir / f"{experiment_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save as pickle (preserves numpy arrays)
        pickle_path = self.output_dir / f"{experiment_name}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Results saved to {json_path} and {pickle_path}")
    
    def load_results(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment results"""
        pickle_path = self.output_dir / f"{experiment_name}_results.pkl"
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
```

**CLAUDE CODE TASK**: Create visualization utilities

**File**: `src/utils/visualization.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

class ExperimentVisualizer:
    """Generate plots for experiment results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['experiment']['output_dir']) / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_risk_vs_dimension(self, results: Dict[str, Any]) -> None:
        """Plot Figure 1 a-d: Risk vs dimension for different μ scalings"""
        dimensions = results['dimensions']
        mu_scalings = results['mu_scalings']
        noise_rate = self.config['data']['noise_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, mu_scaling in enumerate(mu_scalings):
            ax = axes[idx]
            scaling_key = f'mu_scaling_{mu_scaling}'
            scaling_results = results['results'][scaling_key]
            
            # Plot standard and adversarial risks
            ax.plot(dimensions, scaling_results['standard_risks'], 
                   'o-', label='Standard Risk', linewidth=2, markersize=4)
            ax.plot(dimensions, scaling_results['adversarial_risks'], 
                   's-', label='Adversarial Risk', linewidth=2, markersize=4)
            
            # Plot noise level reference
            ax.axhline(y=noise_rate, color='red', linestyle='--', 
                      label=f'Noise Level η={noise_rate}', alpha=0.7)
            
            ax.set_xlabel('Dimension d')
            ax.set_ylabel('Risk')
            ax.set_title(f'μ scaling = d^{mu_scaling}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(0.5, noise_rate * 2))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_vs_dimension.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'risk_vs_dimension.pdf', bbox_inches='tight')
        plt.show()
    
    def plot_risk_vs_iterations(self, results: Dict[str, Any]) -> None:
        """Plot Figure 1 e-f: Risk vs iterations for different ε values"""
        epsilon_values = results['epsilon_values']
        noise_rate = self.config['data']['noise_rate']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for epsilon in epsilon_values:
            epsilon_key = f'epsilon_{epsilon}'
            epsilon_results = results['results'][epsilon_key]
            
            iterations = epsilon_results['iterations']
            
            # Plot standard risk
            ax1.plot(iterations, epsilon_results['standard_risks'], 
                    'o-', label=f'ε={epsilon}', linewidth=2, markersize=3)
            
            # Plot adversarial risk
            ax2.plot(iterations, epsilon_results['adversarial_risks'], 
                    's-', label=f'ε={epsilon}', linewidth=2, markersize=3)
        
        # Add noise level reference
        for ax in [ax1, ax2]:
            ax.axhline(y=noise_rate, color='red', linestyle='--', 
                      label=f'Noise Level η={noise_rate}', alpha=0.7)
            ax.set_xlabel('Training Iterations')
            ax.set_ylabel('Risk')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(0.5, noise_rate * 2))
        
        ax1.set_title('Standard Risk vs Iterations')
        ax2.set_title('Adversarial Risk vs Iterations')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_vs_iterations.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'risk_vs_iterations.pdf', bbox_inches='tight')
        plt.show()
```

**CLAUDE CODE TASK**: Create main execution script

**File**: `run_experiments.py`

```python
#!/usr/bin/env python3
"""
Main entry point for reproducing "Benign Overfitting in Adversarially Robust Linear Classification"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config.config_manager import ConfigManager
from src.experiments.experiment_runner import ExperimentRunner
from src.utils.logging_config import setup_logging

def main():
    parser = argparse.ArgumentParser(
        description='Reproduce benign overfitting experiments'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--experiment', type=str, 
                       choices=['dimension', 'iteration', 'both'],
                       default='both',
                       help='Which experiment to run')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    config_manager = ConfigManager(args.config)
    log_file = Path(config_manager.get('experiment.output_dir')) / 'logs' / 'experiment.log'
    logger = setup_logging(args.log_level, str(log_file))
    
    logger.info("Starting benign overfitting reproduction")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Experiment type: {args.experiment}")
    
    # Run experiments
    runner = ExperimentRunner(config_manager.config)
    
    try:
        if args.experiment in ['dimension', 'both']:
            logger.info("Running dimension experiment...")
            dimension_results = runner.run_dimension_experiment()
            logger.info("Dimension experiment completed successfully")
        
        if args.experiment in ['iteration', 'both']:
            logger.info("Running iteration experiment...")
            iteration_results = runner.run_iteration_experiment()
            logger.info("Iteration experiment completed successfully")
        
        logger.info("All experiments completed successfully!")
        logger.info(f"Results saved in: {config_manager.get('experiment.output_dir')}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
```

**Validation Checkpoint**: Run complete pipeline test

```python
# Create test configuration
test_config = {
    'experiment': {'name': 'test', 'random_seed': 42, 'output_dir': 'test_results'},
    'data': {'n_train': 20, 'n_test': 50, 'noise_rate': 0.1, 'noise_type': 'gaussian'},
    'model': {'initialization': 'xavier_normal'},
    'training': {'learning_rate': 0.01, 'num_iterations': 100, 'p_norm': 2.0, 'epsilon': 0.1},
    'dimension_experiment': {'dimensions': [5, 10], 'mu_scalings': [0.3]},
    'iteration_experiment': {'fixed_dimension': 10, 'mu_scaling': 0.3, 'epsilon_values': [0.05, 0.1]}
}

# Test complete pipeline
runner = ExperimentRunner(test_config)
results = runner.run_dimension_experiment()
assert 'results' in results
print("✅ Phase 4 Complete: Full pipeline validated")
```

## Integration Points and Data Flow

1. **Configuration → Data Generation**: ConfigManager provides parameters to SubGaussianMixtureGenerator
2. **Data Generation → Model Training**: DataLoader output feeds directly into GradientDescentAdversarialTrainer
3. **Model Training → Evaluation**: Trained LinearAdversarialClassifier evaluated by RiskCalculator
4. **Evaluation → Visualization**: Risk metrics processed by ExperimentVisualizer for plotting
5. **Orchestration**: ExperimentRunner coordinates all components and manages results

## Expected Performance and Validation

- **Training Time**: ~5-10 minutes for dimension experiment, ~2-3 minutes for iteration experiment
- **Memory Usage**: <1GB RAM for largest experiments (d=2500)
- **Key Validation Points**:
  - Training error should reach ~0 (perfect fit on noisy training data)
  - Standard risk should approach noise level η=0.1 for sufficient μ scaling
  - Adversarial risk should be higher than standard risk but still approach η
  - Results should match Figure 1 trends from paper

**CLAUDE CODE TASK**: Implement all components following this sequential plan, ensuring each phase validation passes before proceeding to the next phase.

---