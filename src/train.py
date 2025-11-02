"""
Training Script for Fashion-MNIST Classification
This script trains the neural network and logs results to Weights & Biases (wandb)
"""

import tensorflow as tf
import numpy as np
import wandb
import argparse
from tqdm import tqdm
import sys
import os

# Import our custom model and utilities
sys.path.append(os.path.dirname(__file__))

from model import NeuralNetwork, get_optimizer
from utils import load_fashion_mnist, load_mnist, create_batches


def train_model(config, project_name='dl_assignment', use_wandb=True):
    """
    Train neural network with given configuration
    
    Args:
        config: Configuration dictionary with hyperparameters
        project_name: WandB project name
        use_wandb: Whether to use WandB logging
        
    Returns:
        Trained model and test accuracy
    """
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Initialize WandB
    if use_wandb:
        run = wandb.init(
            project=project_name,
            config=config,
            name=f"hl_{config['num_layers']}_hs_{config['hidden_size']}_bs_{config['batch_size']}_opt_{config['optimizer']}_act_{config['activation']}"
        )
        config = wandb.config
    
    # Load data
    print("Loading data...")
    if config.get('dataset', 'fashion_mnist') == 'mnist':
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()
    
    # Create model
    print("Creating model...")
    hidden_sizes = [config['hidden_size']] * config['num_layers']
    
    model = model = NeuralNetwork(
        input_size=784,
        hidden_sizes=hidden_sizes,
        output_size=10,
        activation=config['activation'],
        weight_init=config['weight_init'],
        weight_decay=config.get('weight_decay', 0.0),
        random_seed=42
    )
    
    # Create optimizer
    optimizer = get_optimizer(
        config['optimizer'],
        learning_rate=config['learning_rate']
    )
    
    # Training loop
    print("Starting training...")
    best_val_accuracy = 0.0
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Training phase
        train_losses = []
        train_accuracies = []
        
        train_dataset = create_batches(X_train, y_train, config['batch_size'], shuffle=True)
        
        for X_batch, y_batch in tqdm(train_dataset, desc="Training"):
            loss, accuracy = model.train_step(
                X_batch, 
                y_batch, 
                optimizer,
                loss_type=config.get('loss', 'cross_entropy')
            )
            train_losses.append(float(loss))
            train_accuracies.append(float(accuracy))
        
        train_loss = np.mean(train_losses)
        train_accuracy = np.mean(train_accuracies)
        
        # Validation phase
        val_dataset = create_batches(X_val, y_val, config['batch_size'], shuffle=False)
        val_losses = []
        val_accuracies = []
        
        for X_batch, y_batch in val_dataset:
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_batch, y_pred, config.get('loss', 'cross_entropy'))
            accuracy = model.compute_accuracy(y_batch, y_pred)
            val_losses.append(float(loss))
            val_accuracies.append(float(accuracy))
        
        val_loss = np.mean(val_losses)
        val_accuracy = np.mean(val_accuracies)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if use_wandb:
                model.save(f"models/best_model_{run.id}")
            else:
                model.save(f"models/best_model")
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_dataset = create_batches(X_test, y_test, config['batch_size'], shuffle=False)
    test_losses = []
    test_accuracies = []
    
    for X_batch, y_batch in test_dataset:
        y_pred = model.forward(X_batch)
        loss = model.compute_loss(y_batch, y_pred, config.get('loss', 'cross_entropy'))
        accuracy = model.compute_accuracy(y_batch, y_pred)
        test_losses.append(float(loss))
        test_accuracies.append(float(accuracy))
    
    test_loss = np.mean(test_losses)
    test_accuracy = np.mean(test_accuracies)
    
    if use_wandb:
        wandb.log({
            "test_accuracy": test_accuracy,
            "test_loss": test_loss
        })
        wandb.finish()
    
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return model, test_accuracy, history


def main():
    """Main function for standalone training"""
    parser = argparse.ArgumentParser(description='Train Neural Network')
    
    # Model architecture
    parser.add_argument('--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of hidden layers')
    parser.add_argument('--activation', type=str, default='relu', 
                       choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('--weight_init', type=str, default='xavier', 
                       choices=['random', 'xavier'])
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'momentum', 'nag', 'nesterov', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization')
    parser.add_argument('--loss', type=str, default='cross_entropy', 
                       choices=['cross_entropy', 'mse'])
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                       choices=['fashion_mnist', 'mnist'])
    
    # WandB settings
    parser.add_argument('--project', type=str, default='dl_assignment', help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    
    args = parser.parse_args()
    
    config = {
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'activation': args.activation,
        'weight_init': args.weight_init,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'loss': args.loss,
        'dataset': args.dataset
    }
    
    train_model(config, args.project, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()