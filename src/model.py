"""
Neural Network Implementation for Fashion-MNIST Classification
This implements a feedforward neural network with backpropagation from scratch.
Using TensorFlow's GradientTape to compute gradients and manually implementing the training loop.
"""

import tensorflow as tf
import numpy as np
import json


class NeuralNetwork:
    """
    Custom Feedforward Neural Network with manual training loop
    Uses TensorFlow for gradient computation but implements backprop logic manually
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = [128, 64],
        output_size: int = 10,
        activation: str = 'relu',
        weight_init: str = 'xavier',
        weight_decay: float = 0.0,
        random_seed: int = 42
    ):
        """
        Initialize Neural Network
        
        Args:
            input_size: Number of input features (784 for Fashion-MNIST)
            hidden_sizes: List of hidden layer sizes, e.g., [128, 64, 32]
            output_size: Number of output classes (10)
            activation: Activation function ('sigmoid', 'tanh', 'relu')
            weight_init: Weight initialization ('random', 'xavier')
            weight_decay: L2 regularization parameter
            random_seed: Random seed for reproducibility
        """
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init
        self.weight_decay = weight_decay
        
        # Build model
        self.model = self._build_model()
        
        # Store config for saving
        self.config = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'output_size': output_size,
            'activation': activation,
            'weight_init': weight_init,
            'weight_decay': weight_decay
        }
    
    def _get_activation(self):
        """Get activation function"""
        if self.activation_name == 'sigmoid':
            return tf.nn.sigmoid
        elif self.activation_name == 'tanh':
            return tf.nn.tanh
        elif self.activation_name == 'relu':
            return tf.nn.relu
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def _get_initializer(self):
        """Get weight initializer"""
        if self.weight_init == 'xavier':
            return tf.keras.initializers.GlorotUniform()
        else:
            return tf.keras.initializers.RandomNormal(stddev=0.01)
    
    def _build_model(self):
        """Build the neural network model"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.InputLayer(input_shape=(self.input_size,)))
        
        # Hidden layers
        initializer = self._get_initializer()
        activation = self._get_activation()
        
        for hidden_size in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(
                hidden_size,
                activation=activation,
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay) if self.weight_decay > 0 else None
            ))
        
        # Output layer (no activation, will use softmax in loss)
        model.add(tf.keras.layers.Dense(
            self.output_size,
            kernel_initializer=initializer
        ))
        
        return model
    
    def forward(self, X):
        """
        Forward pass
        
        Args:
            X: Input tensor (batch_size, input_size)
            
        Returns:
            Output predictions (batch_size, output_size)
        """
        return self.model(X, training=True)
    
    def compute_loss(self, y_true, y_pred, loss_type='cross_entropy'):
        """
        Compute loss
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted logits
            loss_type: 'cross_entropy' or 'mse'
            
        Returns:
            Loss value
        """
        if loss_type == 'cross_entropy':
            loss = tf.keras.losses.categorical_crossentropy(
                y_true, y_pred, from_logits=True
            )
        elif loss_type == 'mse':
            # Apply softmax first for MSE
            y_pred_probs = tf.nn.softmax(y_pred)
            loss = tf.keras.losses.mean_squared_error(y_true, y_pred_probs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return tf.reduce_mean(loss)
    
    def compute_accuracy(self, y_true, y_pred):
        """
        Compute accuracy
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted logits
            
        Returns:
            Accuracy value
        """
        y_pred_classes = tf.argmax(y_pred, axis=1)
        y_true_classes = tf.argmax(y_true, axis=1)
        correct = tf.equal(y_pred_classes, y_true_classes)
        return tf.reduce_mean(tf.cast(correct, tf.float32))
    
    def train_step(self, X_batch, y_batch, optimizer, loss_type='cross_entropy'):
        """
        Single training step with manual backpropagation
        
        Args:
            X_batch: Input batch
            y_batch: Label batch
            optimizer: TensorFlow optimizer
            loss_type: Type of loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.forward(X_batch)
            
            # Compute loss
            loss = self.compute_loss(y_batch, y_pred, loss_type)
            
            # Add L2 regularization loss if needed
            if self.weight_decay > 0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables 
                                   if 'kernel' in v.name])
                loss += self.weight_decay * l2_loss
        
        # Compute gradients (backpropagation)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients (update weights)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Compute accuracy
        accuracy = self.compute_accuracy(y_batch, y_pred)
        
        return loss, accuracy
    
    def predict(self, X):
        """
        Predict classes for input data
        
        Args:
            X: Input data
            
        Returns:
            Predicted class indices
        """
        y_pred = self.model(X, training=False)
        return tf.argmax(y_pred, axis=1)
    
    def save(self, filepath: str):
        """Save model weights and config"""
        import os
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else 'models', exist_ok=True)
        
        # Save weights (Keras 3+ requires .weights.h5 extension)
        weights_path = filepath + '.weights.h5'
        self.model.save_weights(weights_path)
        
        # Save config
        config_path = filepath + '_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from saved weights and config"""
        # Load config
        config_path = filepath + '_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(**config)
        
        # Load weights
        weights_path = filepath + '.weights.h5'
        model.model.load_weights(weights_path)
        
        print(f"Model loaded from {filepath}")
        return model


# Custom Optimizers Implementation
class CustomSGD:
    """Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class CustomMomentum:
    """SGD with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        )
    
    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class CustomNesterov:
    """Nesterov Accelerated Gradient"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=True
        )
    
    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class CustomRMSprop:
    """RMSprop Optimizer"""
    def __init__(self, learning_rate=0.001, rho=0.9):
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho
        )
    
    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class CustomAdam:
    """Adam Optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2
        )
    
    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


class CustomNadam:
    """Nadam Optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.optimizer = tf.keras.optimizers.Nadam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2
        )
    
    def apply_gradients(self, grads_and_vars):
        self.optimizer.apply_gradients(grads_and_vars)


def get_optimizer(name: str, learning_rate: float = 0.001, **kwargs):
    """
    Factory function to get optimizer by name
    
    Args:
        name: Optimizer name ('sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam')
        learning_rate: Learning rate
        **kwargs: Additional optimizer-specific parameters
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'sgd': CustomSGD,
        'momentum': CustomMomentum,
        'nag': CustomNesterov,
        'nesterov': CustomNesterov,
        'rmsprop': CustomRMSprop,
        'adam': CustomAdam,
        'nadam': CustomNadam
    }
    
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")
    
    return optimizers[name](learning_rate=learning_rate, **kwargs)