# DATA_AND_EVAL

## Research Context

This documentation enables reproduction of the key empirical findings from "Benign Overfitting in Adversarially Robust Linear Classification" which demonstrates that adversarially trained linear classifiers can achieve zero training error while maintaining good generalization performance. The paper's main contribution is proving that benign overfitting occurs in adversarial settings, not just standard training.

**Target Results to Reproduce**:
- Figure 1 (a-d): Risk vs dimension showing benign overfitting for different μ scalings
- Figure 1 (e-f): Risk vs training iterations for different perturbation budgets ε
- Key finding: Both standard and adversarial risk approach noise level η despite perfect training fit

## Dataset Specifications

### Synthetic Data Generation Model

**Data Source**: Algorithmically generated using Sub-Gaussian mixture model
**No External Downloads Required**: All data generated programmatically

**Data Generation Parameters**:
```python
# Core data model: x̃ = ỹμ + ξ where ỹ ∈ {-1,+1}, ξ ~ sub-Gaussian
DIMENSION_RANGE = [10, 25, 50, 100, 200, 500, 1000, 2500]  # Feature dimensions d
MU_SCALINGS = [0.2, 0.3, 0.4]  # μ norm scaling: ||μ||₂ = d^scaling
N_TRAIN = 50  # Training samples (overparameterized regime: d >> n)
N_TEST = 2000  # Test samples for reliable risk estimation
NOISE_RATE = 0.1  # Label noise η (10% labels flipped)
RANDOM_SEED = 42  # For reproducibility
```

**Data Statistics**:
- **Training Set**: 50 samples × d dimensions (d up to 2500)
- **Test Set**: 2000 samples × d dimensions  
- **Label Distribution**: Balanced binary classification {-1, +1}
- **Feature Distribution**: Sub-Gaussian mixture with controlled separation
- **Noise Characteristics**: 10% label noise in training, clean test labels

**Data Structure**:
```python
# Generated dataset format
dataset = {
    'x_train': np.ndarray,     # Shape: (50, d) - training features
    'y_train': np.ndarray,     # Shape: (50,) - noisy training labels {-1,+1}
    'x_test': np.ndarray,      # Shape: (2000, d) - test features  
    'y_test': np.ndarray,      # Shape: (2000,) - clean test labels {-1,+1}
    'y_train_clean': np.ndarray, # Shape: (50,) - clean training labels
    'y_test_clean': np.ndarray   # Shape: (2000,) - same as y_test
}
```

## Data Processing Pipeline

### Step 1: Mean Vector Generation
```python
def generate_mean_vector(dimension: int, mu_scaling: float) -> np.ndarray:
    """
    Generate separation vector μ with ||μ||₂ = d^mu_scaling
    
    Implementation: μ = (d^mu_scaling / √d) * e₁
    where e₁ is first standard basis vector
    """
    mu = np.zeros(dimension)
    if dimension > 0:
        mu_norm = dimension ** mu_scaling
        mu[0] = mu_norm / np.sqrt(dimension)
    return mu
```

### Step 2: Clean Data Generation  
```python
def generate_clean_samples(n_samples: int, mu: np.ndarray, random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate clean data: x̃ = ỹμ + ξ
    
    Process:
    1. Sample labels ỹ uniformly from {-1, +1}
    2. Sample noise ξ ~ N(0, I) (Gaussian is sub-Gaussian)
    3. Compute features: x̃ᵢ = ỹᵢμ + ξᵢ
    """
    np.random.seed(random_seed)
    
    # Step 1: Generate labels
    y_clean = np.random.choice([-1, 1], size=n_samples)
    
    # Step 2: Generate noise
    xi = np.random.normal(0, 1, size=(n_samples, len(mu)))
    
    # Step 3: Generate features
    x_clean = y_clean.reshape(-1, 1) * mu.reshape(1, -1) + xi
    
    return x_clean, y_clean
```

### Step 3: Label Noise Injection
```python
def inject_label_noise(y_clean: np.ndarray, noise_rate: float, random_seed: int) -> np.ndarray:
    """
    Inject label noise: flip each label with probability η
    
    Process:
    1. Generate flip mask: Bernoulli(η) for each sample
    2. Flip labels: y[flip_mask] = -y[flip_mask]
    """
    np.random.seed(random_seed)
    
    y_noisy = y_clean.copy()
    flip_mask = np.random.random(len(y_clean)) < noise_rate
    y_noisy[flip_mask] = -y_noisy[flip_mask]
    
    return y_noisy
```

### Step 4: Data Validation
```python
def validate_dataset(dataset: Dict[str, np.ndarray]) -> None:
    """Validate generated dataset properties"""
    
    # Shape validation
    n_train, d = dataset['x_train'].shape
    assert dataset['y_train'].shape == (n_train,)
    assert dataset['x_test'].shape[1] == d
    assert np.all(np.isin(dataset['y_train'], [-1, 1]))
    assert np.all(np.isin(dataset['y_test'], [-1, 1]))
    
    # Noise rate validation
    actual_noise_rate = np.mean(dataset['y_train'] != dataset['y_train_clean'])
    expected_noise_rate = 0.1
    assert abs(actual_noise_rate - expected_noise_rate) < 0.05  # Allow 5% tolerance
    
    # Balance validation (should be approximately balanced)
    train_balance = np.mean(dataset['y_train'] == 1)
    test_balance = np.mean(dataset['y_test'] == 1)
    assert 0.3 < train_balance < 0.7  # Reasonable balance range
    assert 0.4 < test_balance < 0.6   # Tighter range for larger test set
```

## Evaluation Framework

### Primary Metrics

#### 1. Standard Risk (Clean Test Error)
```python
def compute_standard_risk(model: LinearAdversarialClassifier, 
                         x_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Standard Risk = P[f(x) ≠ y] on clean test data
    
    Formula: (1/n_test) * Σᵢ 𝟙[sign(θᵀxᵢ) ≠ yᵢ]
    Expected Range: [0.1, 0.5] (should approach η=0.1 for good models)
    """
    predictions = model.predict(x_test)  # sign(θᵀx)
    return np.mean(predictions != y_test)
```

#### 2. Adversarial Risk (Robust Test Error)
```python
def compute_adversarial_risk(model: LinearAdversarialClassifier,
                           x_test: np.ndarray, y_test: np.ndarray,
                           p_norm: float, epsilon: float) -> float:
    """
    Adversarial Risk = P[∃x' ∈ Bₚᵋ(x) s.t. f(x') ≠ y]
    
    For linear models: Adversarial error when yᵢθᵀxᵢ ≤ ε||θ||_q
    where q is dual of p (1/p + 1/q = 1)
    
    Expected Range: [0.1, 0.6] (higher than standard risk)
    """
    theta = model.get_parameters()
    
    # Compute margins and dual norm
    margins = y_test * (x_test @ theta)
    dual_norm = compute_dual_norm(theta, p_norm)
    
    # Adversarial margin: yᵢθᵀxᵢ - ε||θ||_q
    adversarial_margins = margins - epsilon * dual_norm
    
    # Error when adversarial margin ≤ 0
    return np.mean(adversarial_margins <= 0)
```

#### 3. Training Error (Overfitting Indicator)
```python
def compute_training_error(model: LinearAdversarialClassifier,
                          x_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Training Error = Classification error on training set
    
    Expected Behavior: Should approach 0 (perfect fit despite noise)
    This is the "overfitting" part of "benign overfitting"
    """
    predictions = model.predict(x_train)
    return np.mean(predictions != y_train)
```

### Secondary Metrics

#### 4. Parameter Norm Growth
```python
def track_parameter_norm(model: LinearAdversarialClassifier) -> float:
    """
    Track ||θ||₂ growth during training
    
    Expected Behavior: Should grow during training but stabilize
    Important for understanding convergence behavior
    """
    return np.linalg.norm(model.get_parameters(), ord=2)
```

#### 5. Loss Function Value
```python
def compute_adversarial_loss(model: LinearAdversarialClassifier,
                           x: np.ndarray, y: np.ndarray,
                           p_norm: float, epsilon: float) -> float:
    """
    Adversarial exponential loss: Σᵢ exp(-yᵢθᵀxᵢ + ε||θ||_q)
    
    Expected Behavior: Should decrease during training
    """
    theta = model.get_parameters()
    margins = y * (x @ theta)
    dual_norm = compute_dual_norm(theta, p_norm)
    
    losses = np.exp(-margins + epsilon * dual_norm)
    return np.mean(losses)
```

## Experimental Protocols

### Experiment 1: Risk vs Dimension

**Purpose**: Demonstrate benign overfitting as dimension increases

**Parameters**:
```python
EXPERIMENT_1_CONFIG = {
    'dimensions': [10, 25, 50, 100, 200, 500, 1000, 2500],
    'mu_scalings': [0.2, 0.3, 0.4],  # Different growth rates for ||μ||₂
    'fixed_params': {
        'n_train': 50,
        'n_test': 2000,
        'noise_rate': 0.1,
        'learning_rate': 0.001,
        'num_iterations': 1000,
        'p_norm': 2.0,  # L₂ perturbations
        'epsilon': 0.1
    }
}
```

**Procedure**:
1. For each μ scaling in [0.2, 0.3, 0.4]:
   2. For each dimension d in [10, 25, ..., 2500]:
      3. Generate dataset with ||μ||₂ = d^scaling
      4. Train adversarial linear classifier for 1000 iterations
      5. Compute standard risk, adversarial risk, training error
      6. Record final values

**Expected Results**:
- **μ scaling = 0.2**: Risks remain high (no benign overfitting)
- **μ scaling = 0.3**: Risks decrease with dimension but slowly
- **μ scaling = 0.4**: Clear benign overfitting - risks approach η=0.1

### Experiment 2: Risk vs Training Iterations

**Purpose**: Show convergence behavior for different perturbation budgets

**Parameters**:
```python
EXPERIMENT_2_CONFIG = {
    'fixed_dimension': 200,
    'mu_scaling': 0.3,
    'epsilon_values': [0.05, 0.1, 0.2],  # Different perturbation budgets
    'evaluation_points': 50,  # Number of evaluation points during training
    'fixed_params': {
        'n_train': 50,
        'n_test': 2000,
        'noise_rate': 0.1,
        'learning_rate': 0.001,
        'num_iterations': 1000,
        'p_norm': 2.0
    }
}
```

**Procedure**:
1. Generate fixed dataset with d=200, ||μ||₂ = 200^0.3
2. For each ε in [0.05, 0.1, 0.2]:
   3. Initialize fresh model
   4. Train for 1000 iterations, evaluating every 20 iterations
   5. Record risk evolution over training

**Expected Results**:
- All models should achieve near-zero training error
- Standard risk should be similar across ε values
- Adversarial risk should increase with ε
- Both risks should approach η=0.1 by end of training

### Hyperparameter Settings from Paper

**Training Configuration**:
```python
PAPER_HYPERPARAMETERS = {
    'initialization': 'xavier_normal',  # Paper uses Xavier, not zeros
    'learning_rate': 0.001,  # Fixed step size α
    'num_iterations': 1000,  # Sufficient for convergence
    'p_norm': 2.0,  # L₂ adversarial perturbations (also test L∞)
    'epsilon_l2': 0.1,  # L₂ perturbation budget
    'epsilon_linf': 0.01,  # L∞ perturbation budget (smaller due to different scaling)
}
```

**Data Configuration**:
```python
PAPER_DATA_CONFIG = {
    'n_train': 50,  # Small for overparameterization
    'n_test': 2000,  # Large for reliable risk estimation
    'noise_rate': 0.1,  # 10% label noise
    'noise_type': 'gaussian',  # Standard Gaussian noise ξ ~ N(0,I)
    'random_seed': 42,  # For reproducibility
}
```

## Results Validation

### Expected Performance Ranges

**Dimension Experiment (Figure 1 a-d)**:
```python
EXPECTED_RANGES = {
    'mu_scaling_0.2': {
        'standard_risk': [0.25, 0.45],  # Should remain high
        'adversarial_risk': [0.35, 0.55],
        'training_error': [0.0, 0.05]  # Should be near zero
    },
    'mu_scaling_0.3': {
        'standard_risk': [0.15, 0.35],  # Moderate improvement
        'adversarial_risk': [0.25, 0.45],
        'training_error': [0.0, 0.05]
    },
    'mu_scaling_0.4': {
        'standard_risk': [0.10, 0.20],  # Clear benign overfitting
        'adversarial_risk': [0.15, 0.30],
        'training_error': [0.0, 0.05]
    }
}
```

**Iteration Experiment (Figure 1 e-f)**:
```python
EXPECTED_CONVERGENCE = {
    'final_standard_risk': [0.10, 0.20],  # Should approach η=0.1
    'final_adversarial_risk': {
        'epsilon_0.05': [0.12, 0.22],
        'epsilon_0.1': [0.15, 0.25],
        'epsilon_0.2': [0.20, 0.35]
    },
    'convergence_iterations': [200, 800],  # Should converge within 1000 iterations
}
```

### Automated Validation Scripts

```python
def validate_dimension_results(results: Dict[str, Any]) -> bool:
    """Validate dimension experiment results match expected patterns"""
    
    for mu_scaling in [0.2, 0.3, 0.4]:
        scaling_key = f'mu_scaling_{mu_scaling}'
        scaling_results = results['results'][scaling_key]
        
        # Check that risks generally decrease with dimension for higher scalings
        if mu_scaling >= 0.3:
            std_risks = scaling_results['standard_risks']
            # Risk at highest dimension should be lower than at lowest
            assert std_risks[-1] < std_risks[0], f"No improvement for μ scaling {mu_scaling}"
        
        # Training error should be near zero for all cases
        train_errors = scaling_results['training_errors']
        assert all(err < 0.1 for err in train_errors), "Training error too high"
        
        # Adversarial risk should be higher than standard risk
        std_risks = scaling_results['standard_risks']
        adv_risks = scaling_results['adversarial_risks']
        assert all(adv >= std for adv, std in zip(adv_risks, std_risks)), \
               "Adversarial risk should exceed standard risk"
    
    return True

def validate_iteration_results(results: Dict[str, Any]) -> bool:
    """Validate iteration experiment results match expected patterns"""
    
    for epsilon in results['epsilon_values']:
        epsilon_key = f'epsilon_{epsilon}'
        epsilon_results = results['results'][epsilon_key]
        
        # Risks should generally decrease over iterations
        std_risks = epsilon_results['standard_risks']
        adv_risks = epsilon_results['adversarial_risks']
        
        # Final risk should be lower than initial risk
        assert std_risks[-1] < std_risks[0], f"Standard risk didn't improve for ε={epsilon}"
        assert adv_risks[-1] < adv_risks[0], f"Adversarial risk didn't improve for ε={epsilon}"
        
        # Training error should approach zero
        train_errors = epsilon_results['training_errors']
        assert train_errors[-1] < 0.1, f"Training error too high for ε={epsilon}"
    
    return True
```

### Visualization and Reporting

**Required Plots**:
1. **Figure 1a-d**: Risk vs dimension for different μ scalings (2×2 subplot)
2. **Figure 1e-f**: Risk vs iterations for different ε values (1×2 subplot)

**Plot Specifications**:
```python
PLOT_CONFIG = {
    'figure_size': (12, 10),  # For dimension plots
    'line_styles': {
        'standard_risk': 'o-',
        'adversarial_risk': 's-',
        'noise_level': '--'
    },
    'colors': ['blue', 'red', 'green', 'orange'],
    'save_formats': ['png', 'pdf'],
    'dpi': 300,
    'include_noise_reference': True,  # Show η=0.1 line
}
```

## Reproducibility Requirements

### Random Seed Management
```python
REPRODUCIBILITY_CONFIG = {
    'global_seed': 42,
    'train_data_seed': 42,
    'test_data_seed': 1042,  # Different seed to avoid train/test overlap
    'model_init_seed': 2042,
    'numpy_seed_before_each_experiment': True
}
```

### Version Pinning Strategy
```python
EXACT_VERSIONS = {
    'python': '3.9+',
    'numpy': '1.24.3',
    'scipy': '1.10.1',
    'matplotlib': '3.7.1',
    'seaborn': '0.12.2'
}
```

### Environment Configuration
```python
ENVIRONMENT_SETTINGS = {
    'numpy_random_seed': 42,
    'numpy_print_precision': 6,
    'matplotlib_backend': 'Agg',  # For headless environments
    'warning_filters': ['ignore::RuntimeWarning'],  # Suppress numerical warnings
}
```

### Expected Output Structure
```
results/
├── dimension_experiment_results.json    # Human-readable results
├── dimension_experiment_results.pkl     # Full numpy arrays
├── iteration_experiment_results.json
├── iteration_experiment_results.pkl
├── figures/
│   ├── risk_vs_dimension.png
│   ├── risk_vs_dimension.pdf
│   ├── risk_vs_iterations.png
│   └── risk_vs_iterations.pdf
└── logs/
    └── experiment.log                    # Detailed execution log
```

### Performance Benchmarks
- **Total Runtime**: 15-30 minutes for both experiments
- **Memory Usage**: <2GB peak (for d=2500 experiments)
- **Disk Usage**: <100MB for all results and figures
- **CPU Usage**: Single-threaded (no parallelization needed)

**Success Criteria**: Generated figures should qualitatively match Figure 1 from the paper, showing clear benign overfitting trends for appropriate parameter settings.

---