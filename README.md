# Deep Learning Assignment: Feedforward Neural Network with Backpropagation

Implementation of a feedforward neural network from scratch for Fashion-MNIST classification, demonstrating backpropagation algorithm and comprehensive hyperparameter tuning using Weights & Biases.

---

## Table of Contents

1. [Overview](#overview)
2. [Technical Architecture](#technical-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Project Structure](#project-structure)
5. [Implementation Details](#implementation-details)
6. [Usage Instructions](#usage-instructions)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Results and Analysis](#results-and-analysis)
9. [Technical Terminology](#technical-terminology)
10. [Dependencies](#dependencies)
11. [My WandB Report](#my-wandb-report)

---

## Overview

This project implements a feedforward neural network with manual backpropagation training loop for multi-class image classification. The implementation uses TensorFlow's automatic differentiation capabilities while maintaining explicit control over the training process to demonstrate understanding of backpropagation mechanics.

### Problem Statement

Classify 28x28 grayscale images from the Fashion-MNIST dataset into 10 clothing categories using a fully-connected neural network trained via backpropagation.

### Dataset: Fashion-MNIST

Fashion-MNIST is a dataset of Zalando's article images consisting of:
- Training set: 60,000 examples
- Test set: 10,000 examples
- Image size: 28x28 grayscale pixels
- Classes: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

The dataset serves as a direct drop-in replacement for MNIST but is more challenging due to the complexity of clothing items compared to handwritten digits.

---

## Technical Architecture

### Neural Network Architecture

**Model Type:** Feedforward Neural Network (Multilayer Perceptron)

**Architecture Components:**
- Input Layer: 784 neurons (28x28 flattened image)
- Hidden Layers: Configurable (default: 3 layers)
- Hidden Layer Size: Configurable (tested: 32, 64, 128 neurons)
- Output Layer: 10 neurons (one per class)
- Activation Functions: Sigmoid, Tanh, or ReLU
- Output Activation: Softmax (implicit in loss function)

**Forward Propagation:**
```
Input (784) → Dense(n) → Activation → Dense(n) → Activation → ... → Dense(10) → Softmax → Output
```

**Backpropagation:**
The training process computes gradients using automatic differentiation (TensorFlow GradientTape) and updates weights using various optimization algorithms.

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum
- Internet connection for dataset download

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/dl_assignment.git
cd dl_assignment
```

### Step 2: Create Virtual Environment

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including TensorFlow, NumPy, Matplotlib, Weights & Biases, etc.

### Step 4: Setup Weights & Biases

```bash
wandb login
```

Enter your API key from https://wandb.ai/authorize when prompted.

### Step 5: Verify Installation

```bash
python -c "import tensorflow as tf; from src.model import NeuralNetwork; print('Installation successful')"
```

---

## Project Structure

```
dl_assignment/
├── src/
│   ├── __init__.py              # Package initializer
│   ├── model.py                 # Neural network and optimizer implementations
│   ├── train.py                 # Training script with WandB integration
│   ├── evaluate.py              # Evaluation and visualization script
│   └── utils.py                 # Data loading and utility functions
│
├── config/
│   └── sweep_config.yaml        # WandB sweep configuration
│
├── models/                      # Saved model weights (generated)
│   └── best_model_*.weights.h5
│
├── plots/                       # Generated visualizations (generated)
│   ├── confusion_matrix_*.png
│   └── loss_comparison.png
│
├── run_sweep.py                 # Hyperparameter sweep orchestration
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore patterns
```

### File Descriptions

**src/model.py:**
Contains the core neural network implementation including:
- `NeuralNetwork` class: Implements forward propagation, loss computation, and training step
- Optimizer classes: SGD, Momentum, Nesterov, RMSProp, Adam, NAdam
- `get_optimizer()`: Factory function for optimizer selection

**src/train.py:**
Training orchestration script that:
- Loads and preprocesses data
- Creates model with specified architecture
- Executes training loop with mini-batch gradient descent
- Logs metrics to Weights & Biases
- Saves best model based on validation accuracy

**src/evaluate.py:**
Evaluation script providing:
- Model performance assessment on test set
- Confusion matrix generation
- Per-class accuracy analysis
- Loss function comparison (cross-entropy vs MSE)
- MNIST transfer learning experiments

**src/utils.py:**
Utility functions for:
- Dataset loading (Fashion-MNIST, MNIST)
- Data preprocessing and normalization
- Batch creation for mini-batch training
- Visualization functions (confusion matrix, accuracy plots, sample predictions)

**config/sweep_config.yaml:**
Hyperparameter search configuration specifying:
- Search method (Bayesian optimization)
- Parameter ranges
- Optimization metric
- Early termination strategy

**run_sweep.py:**
Orchestrates hyperparameter sweep by:
- Loading sweep configuration
- Creating WandB sweep
- Managing sweep agents
- Coordinating multiple parallel runs

---

## Implementation Details

### Backpropagation Implementation

The backpropagation algorithm is implemented using TensorFlow's automatic differentiation:

```python
def train_step(self, X_batch, y_batch, optimizer, loss_type='cross_entropy'):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = self.forward(X_batch)
        
        # Compute loss
        loss = self.compute_loss(y_batch, predictions, loss_type)
        
        # Add L2 regularization
        if self.weight_decay > 0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables 
                               if 'kernel' in v.name])
            loss += self.weight_decay * l2_loss
    
    # Backpropagation: compute gradients
    gradients = tape.gradient(loss, self.model.trainable_variables)
    
    # Update weights using optimizer
    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    return loss, accuracy
```

**Key Steps:**
1. Forward pass computes predictions
2. Loss function measures prediction error
3. GradientTape records operations for differentiation
4. gradient() computes partial derivatives via chain rule
5. Optimizer updates weights using computed gradients

### Optimizer Implementations

**1. Stochastic Gradient Descent (SGD):**
Basic gradient descent with fixed learning rate.
```
θ = θ - η∇L(θ)
```
Where θ is parameters, η is learning rate, ∇L is gradient.

**2. Momentum:**
Accelerates SGD by accumulating gradient history.
```
v = βv + ∇L(θ)
θ = θ - ηv
```
Where v is velocity, β is momentum coefficient (typically 0.9).

**3. Nesterov Accelerated Gradient (NAG):**
Look-ahead version of momentum that evaluates gradient at anticipated future position.

**4. RMSProp:**
Adapts learning rate for each parameter based on running average of squared gradients.
```
E[g²] = βE[g²] + (1-β)g²
θ = θ - η * g / √(E[g²] + ε)
```

**5. Adam (Adaptive Moment Estimation):**
Combines momentum and RMSProp by maintaining running averages of both gradient and squared gradient.
```
m = β₁m + (1-β₁)g
v = β₂v + (1-β₂)g²
θ = θ - η * m̂ / (√v̂ + ε)
```

**6. NAdam:**
Combines Nesterov momentum with Adam's adaptive learning rates.

### Loss Functions

**Cross-Entropy Loss (Primary):**
Measures divergence between predicted probability distribution and true distribution.
```
L = -∑ y_true * log(y_pred)
```
Optimal for classification tasks with probabilistic outputs.

**Mean Squared Error (Comparison):**
Measures squared difference between predictions and targets.
```
L = (1/n)∑(y_true - y_pred)²
```
Typically used for regression; suboptimal for classification.

### Regularization

**L2 Weight Decay:**
Adds penalty term to loss function to prevent overfitting:
```
L_total = L_data + λ∑w²
```
Where λ is regularization strength (weight_decay parameter).

---

## Usage Instructions

### Training a Single Model

**Basic Training:**
```bash
cd src
python train.py --epochs 10 --optimizer adam --batch_size 32
```

**Advanced Training with All Parameters:**
```bash
python train.py \
  --num_layers 3 \
  --hidden_size 128 \
  --activation relu \
  --optimizer adam \
  --learning_rate 0.001 \
  --batch_size 32 \
  --epochs 10 \
  --weight_decay 0.0005 \
  --weight_init xavier \
  --loss cross_entropy \
  --dataset fashion_mnist
```

**Training Without WandB (Local Testing):**
```bash
python train.py --epochs 5 --batch_size 64 --no_wandb
```

### Command-Line Arguments

| Argument | Type | Default | Options | Description |
|----------|------|---------|---------|-------------|
| `--num_layers` | int | 3 | Any positive int | Number of hidden layers |
| `--hidden_size` | int | 128 | Any positive int | Neurons per hidden layer |
| `--activation` | str | relu | sigmoid, tanh, relu | Activation function |
| `--weight_init` | str | xavier | random, xavier | Weight initialization method |
| `--optimizer` | str | adam | sgd, momentum, nag, rmsprop, adam, nadam | Optimization algorithm |
| `--learning_rate` | float | 0.001 | Any positive float | Learning rate for optimizer |
| `--batch_size` | int | 32 | Any positive int | Mini-batch size |
| `--epochs` | int | 10 | Any positive int | Number of training epochs |
| `--weight_decay` | float | 0.0 | Non-negative float | L2 regularization strength |
| `--loss` | str | cross_entropy | cross_entropy, mse | Loss function |
| `--dataset` | str | fashion_mnist | fashion_mnist, mnist | Dataset selection |
| `--no_wandb` | flag | False | - | Disable WandB logging |

### Evaluating Trained Model

```bash
python evaluate.py --model_path ../models/best_model_XXXXX --dataset fashion_mnist
```

Replace `XXXXX` with the actual run ID from your training output.

**Outputs:**
- Test accuracy and loss metrics
- Confusion matrix (both count and normalized)
- Per-class accuracy bar chart
- Sample prediction visualizations

### Comparing Loss Functions

```bash
python evaluate.py --compare_losses
```

Trains two identical models with different loss functions and compares performance.

### MNIST Transfer Learning

```bash
python evaluate.py --mnist_configs
```

Tests three configurations on MNIST dataset based on Fashion-MNIST learnings.

---

## Hyperparameter Tuning

### Sweep Configuration

The hyperparameter search uses Bayesian optimization to efficiently explore the parameter space.

**Search Space:**
- Epochs: [5, 8]
- Hidden Layers: [3]
- Layer Size: [64, 128]
- Weight Decay: [0, 0.0005]
- Learning Rate: [0.001, 0.0001]
- Optimizers: [sgd, momentum, rmsprop, adam, nadam]
- Batch Size: [32, 64]
- Activation: [relu, tanh]

**Total Combinations:** 160 possible configurations  
**Actual Runs:** 20 (Bayesian optimization selects most promising)

### Running Hyperparameter Sweep

**Option 1: Automated Sweep**
```bash
python run_sweep.py --count 20
```

**Option 2: Manual Sweep Control**
```bash
# Create sweep
python run_sweep.py --create_only

# Run agents (can run multiple in parallel)
wandb agent YOUR_USERNAME/dl_assignment/SWEEP_ID
```

**Parallel Execution:**
Open multiple terminals and run the agent command in each for faster completion.

### Sweep Strategy

**Bayesian Optimization:**
- Uses probabilistic model to predict promising configurations
- Balances exploration (trying new areas) and exploitation (refining good areas)
- More efficient than random or grid search

**Early Termination (Hyperband):**
- Stops poorly performing runs after few epochs
- Saves computational resources
- Focuses budget on promising configurations

---

## Results and Analysis

### Expected Performance

**Fashion-MNIST:**
- Baseline (SGD, no tuning): 75-80% accuracy
- Good configuration: 85-88% accuracy
- Best configuration: 88-90% accuracy

**MNIST:**
- Good configuration: 96-97% accuracy
- Best configuration: 97-98% accuracy

### Key Findings

**Best Configuration:**
- Architecture: 3 hidden layers × 128 neurons
- Optimizer: Adam (learning_rate=0.001)
- Activation: ReLU
- Weight Init: Xavier
- Batch Size: 64 (optimal for CPU training)
- Weight Decay: 0.0005

**Important Observations:**
1. Adam optimizer consistently outperforms SGD-based methods
2. ReLU activation superior to sigmoid/tanh (avoids vanishing gradients)
3. Xavier initialization critical for deep networks
4. Small weight decay improves generalization
5. Larger batch sizes beneficial for CPU training efficiency

### Confusion Matrix Insights

Common misclassifications occur between visually similar items:
- Shirt ↔ T-shirt/top
- Pullover ↔ Coat
- Sneaker ↔ Ankle boot

Best recognized classes:
- Trouser (distinct shape)
- Bag (unique category)
- Ankle boot (distinctive features)

---

## Technical Terminology

### Core Concepts

**Epoch:**
One complete pass through the entire training dataset. Multiple epochs allow the model to see data multiple times and learn patterns progressively.

**Batch Size:**
Number of training examples processed together before updating weights. Larger batches provide more stable gradients but require more memory.

**Mini-batch Gradient Descent:**
Training approach that updates weights after processing a small batch of examples, balancing computational efficiency and convergence stability.

**Learning Rate:**
Step size for weight updates during optimization. Controls how much weights change in response to gradient. Too high causes instability, too low causes slow convergence.

**Activation Function:**
Non-linear transformation applied to neuron outputs. Introduces non-linearity allowing network to learn complex patterns.

**Backpropagation:**
Algorithm for computing gradients of loss function with respect to weights. Uses chain rule to propagate error backwards through network layers.

**Gradient Descent:**
Optimization algorithm that iteratively adjusts parameters in direction opposite to gradient to minimize loss function.

**Forward Propagation:**
Process of computing network output by passing input through layers sequentially.

**Loss Function:**
Mathematical function measuring difference between predictions and true labels. Training aims to minimize this value.

**Validation Set:**
Subset of training data held out for hyperparameter tuning and model selection without contaminating test set.

**Overfitting:**
Phenomenon where model learns training data too well, including noise, resulting in poor generalization to new data.

**Regularization:**
Techniques to prevent overfitting by constraining model complexity (e.g., L2 weight decay).

**One-Hot Encoding:**
Representation of categorical labels as binary vectors (e.g., class 3 → [0,0,0,1,0,0,0,0,0,0]).

**Softmax:**
Function converting logits (raw outputs) to probability distribution over classes.

### Advanced Concepts

**Xavier/Glorot Initialization:**
Weight initialization method that scales initial weights based on layer sizes to maintain stable gradients during backpropagation.

**Vanishing Gradient Problem:**
Issue in deep networks where gradients become extremely small, preventing effective learning in early layers. Addressed by using ReLU and proper initialization.

**Hyperparameter:**
Configuration parameter set before training (e.g., learning rate, number of layers) as opposed to learned parameters (weights, biases).

**Bayesian Optimization:**
Sequential model-based optimization that uses probabilistic models to guide hyperparameter search toward promising regions.

**Cross-Entropy Loss:**
Loss function measuring difference between predicted and true probability distributions. Mathematically equivalent to negative log-likelihood.

**Gradient Tape (TensorFlow):**
Context manager that records operations for automatic differentiation, enabling gradient computation.

**Momentum:**
Optimization technique that accumulates gradient history to accelerate convergence and dampen oscillations.

**Adaptive Learning Rate:**
Techniques (RMSProp, Adam) that adjust learning rate per parameter based on gradient history.

---

## Dependencies

### Core Libraries

**TensorFlow (>=2.10.0):**
Deep learning framework providing:
- Automatic differentiation via GradientTape
- Optimized tensor operations
- GPU acceleration support
- Pre-built layers and activation functions

**NumPy (>=1.21.0):**
Numerical computing library for:
- Array operations
- Mathematical functions
- Data manipulation

**Pandas (>=1.3.0):**
Data manipulation library (used for potential data analysis extensions).

### Visualization Libraries

**Matplotlib (>=3.4.0):**
Plotting library for:
- Training curves
- Loss visualization
- Custom plots

**Seaborn (>=0.11.0):**
Statistical visualization library built on Matplotlib:
- Confusion matrix heatmaps
- Enhanced plot aesthetics

### Machine Learning Utilities

**scikit-learn (>=1.0.0):**
Machine learning utilities:
- Confusion matrix computation
- Performance metrics
- Data preprocessing utilities

### Experiment Tracking

**Weights & Biases (>=0.15.0):**
Experiment tracking platform providing:
- Metric logging and visualization
- Hyperparameter sweep orchestration
- Model versioning
- Collaborative experiment sharing
- Parallel coordinates plots
- Run comparison tools

### Supporting Libraries

**tqdm (>=4.65.0):**
Progress bar library for training loop visualization.

**PyYAML (>=6.0):**
YAML parser for configuration file handling.

### Complete requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
tensorflow>=2.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
wandb>=0.15.0
tqdm>=4.65.0
PyYAML>=6.0
```

---

##  My WandB Report

### Report Link

[WandB Report URL]


### Report Contents

The complete WandB report includes:

**1. Implementation Overview**
- Architecture description
- Backpropagation implementation approach
- Link to GitHub repository

**2. Hyperparameter Search Results**
- Sweep configuration and strategy
- Total runs completed
- Parameter ranges explored
- Bayesian optimization effectiveness

**3. Best Model Performance**
- Highest validation accuracy achieved
- Optimal hyperparameter configuration
- Training and validation curves
- Convergence analysis

**4. Experimental Analysis**
- Parallel coordinates plot analysis
- Hyperparameter importance rankings
- Performance patterns and trends
- Optimizer comparisons
- Activation function effectiveness
- Impact of regularization

**5. Test Set Evaluation**
- Final test accuracy and loss
- Confusion matrix (count and normalized)
- Per-class accuracy breakdown
- Misclassification patterns
- Sample predictions with visualizations

**6. Loss Function Comparison**
- Cross-entropy vs MSE performance
- Training curves comparison
- Convergence speed analysis
- Final accuracy comparison
- Theoretical justification

**7. Code Repository**
- GitHub repository link
- Commit history overview
- Code structure explanation

**8. MNIST Transfer Learning**
- Three configuration results
- Performance comparison
- Accuracy metrics
- Insights on dataset difficulty

---


### Environment

Python version and library versions specified in requirements.txt ensure consistent behavior across different systems.


---

## Known Limitations

1. **Computational Constraints:**  This Wandb report experiments conducted on i3 laptop with 8GB RAM limited sweep size to 20 runs instead of typical 50-100 runs.

2. **Batch Size Selection:** Larger batch sizes (64) preferred over smaller (16, 32) to optimize CPU training efficiency.

3. **Epoch Limitation:** Reduced maximum epochs to 8 for faster iteration during sweep.

4. **Parameter Space:** Focused on most impactful hyperparameters, excluding some combinations for computational efficiency.

---

## References

1. Fashion-MNIST Dataset: Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.

2. Adam Optimizer: Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.

3. Batch Normalization: Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training.

4. Xavier Initialization: Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.







