"""
Run WandB Sweep for Hyperparameter Optimization
"""

import wandb
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import train_model


def train_sweep():
    """Run a single sweep training"""
    # Initialize WandB run
    run = wandb.init()
    
    # Get config from sweep
    config = dict(wandb.config)
    
    print(f"\n{'='*70}")
    print(f"Running configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print('='*70)
    
    # Train model
    train_model(config, project_name=run.project, use_wandb=True)


def main():
    """Main function to initialize and run sweep"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run WandB Sweep')
    parser.add_argument('--config', type=str, default='config/sweep_config.yaml',
                       help='Path to sweep config file')
    parser.add_argument('--count', type=int, default=50,
                       help='Number of runs for this agent')
    parser.add_argument('--project', type=str, default='dl_assignment',
                       help='WandB project name')
    parser.add_argument('--create_only', action='store_true',
                       help='Only create sweep, don\'t run agent')
    
    args = parser.parse_args()
    
    # Load sweep configuration
    with open(args.config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project
    )
    
    print(f"\n{'='*70}")
    print(f" Created sweep with ID: {sweep_id}")
    print(f"{'='*70}")
    print(f"\nTo run this sweep:")
    print(f"  wandb agent {sweep_id}")
    print(f"\nOr run multiple agents in parallel:")
    print(f"  # Terminal 1:")
    print(f"  wandb agent {sweep_id}")
    print(f"  # Terminal 2:")
    print(f"  wandb agent {sweep_id}")
    print(f"  # Terminal 3:")
    print(f"  wandb agent {sweep_id}")
    print(f"\n{'='*70}\n")
    
    if not args.create_only:
        print(f"Starting agent with {args.count} runs...")
        # Run sweep agent
        wandb.agent(sweep_id, function=train_sweep, count=args.count)
        print("\n Sweep agent completed!")


if __name__ == "__main__":
    main()