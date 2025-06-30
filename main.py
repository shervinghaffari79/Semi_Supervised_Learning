"""
Main script for running SSL experiments.
Demonstrates the organized, class-based configuration system.
"""

import torch
import os
from typing import Dict, Any

from config import ExperimentConfig, get_full_experiment_config, get_baseline_config, get_small_experiment_config
from data_utils import DataManager
from models import create_autoencoder, create_classifier, print_model_summary
from trainers import AutoencoderTrainer, ClassifierTrainer, evaluate_model
from utils import set_seed, setup_device, create_directories, print_experiment_summary, visualize_features_tsne


def run_ssl_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a complete SSL experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary containing all results
    """
    # Validate configuration
    config.validate()
    
    # Setup reproducibility
    set_seed(config.data.random_seed)
    
    # Setup device
    device = config.get_device()
    print(f"Using device: {device}")
    
    # Create necessary directories
    create_directories([config.training.checkpoint_dir])
    
    # Setup data
    print("\n=== Setting up data ===")
    data_manager = DataManager(config.data)
    data_manager.setup_datasets()
    data_manager.print_data_summary()
    
    # Get data loaders
    labeled_loader, unlabeled_loader, val_loader, test_loader = data_manager.get_data_loaders()
    
    results = {}
    
    # Phase 1: Autoencoder pre-training (if enabled)
    if config.training.autoencoder_epochs > 0:
        print("\n=== Phase 1: Autoencoder Pre-training ===")
        
        # Create autoencoder
        autoencoder = create_autoencoder(config.model).to(device)
        print_model_summary(autoencoder, "Autoencoder")
        
        # Train autoencoder
        autoencoder_trainer = AutoencoderTrainer(
            autoencoder, 
            config.training, 
            device,
            config.training.checkpoint_dir
        )
        
        autoencoder_results = autoencoder_trainer.train(unlabeled_loader, val_loader)
        results['autoencoder'] = autoencoder_results
        
        # Visualize learned features
        print("\n=== Visualizing learned features ===")
        visualize_features_tsne(
            autoencoder, 
            val_loader, 
            device, 
            max_samples=2000,
            save_path=os.path.join(config.training.checkpoint_dir, "tsne_features.png")
        )
        
        # Get pre-trained encoder
        pretrained_encoder = autoencoder.encoder
    else:
        print("\n=== Skipping autoencoder pre-training ===")
        pretrained_encoder = None
        results['autoencoder'] = None
    
    # Phase 2: Classifier fine-tuning
    print("\n=== Phase 2: Classifier Training ===")
    
    # Create classifier
    classifier = create_classifier(config.model, pretrained_encoder).to(device)
    print_model_summary(classifier, "Classifier")
    
    # Train classifier
    classifier_trainer = ClassifierTrainer(
        classifier, 
        config.training, 
        device,
        config.training.checkpoint_dir
    )
    
    classifier_results = classifier_trainer.train(labeled_loader, val_loader)
    results['classifier'] = classifier_results
    
    # Phase 3: Final evaluation
    print("\n=== Phase 3: Final Evaluation ===")
    test_accuracy = evaluate_model(classifier, test_loader, device)
    results['test_accuracy'] = test_accuracy
    
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    
    # Print experiment summary
    print_experiment_summary(config, {
        'Best Validation Accuracy': classifier_results['best_val_acc'],
        'Test Accuracy': test_accuracy
    })
    
    return results


def run_baseline_experiment() -> Dict[str, Any]:
    """Run baseline experiment (no pre-training)."""
    print("Running BASELINE experiment (no pre-training)")
    config = get_baseline_config()
    return run_ssl_experiment(config)


def run_full_ssl_experiment() -> Dict[str, Any]:
    """Run full SSL experiment with pre-training."""
    print("Running FULL SSL experiment (with pre-training)")
    config = get_full_experiment_config()
    return run_ssl_experiment(config)


def run_small_experiment() -> Dict[str, Any]:
    """Run small experiment for quick testing."""
    print("Running SMALL experiment (for testing)")
    config = get_small_experiment_config()
    return run_ssl_experiment(config)


def compare_experiments() -> None:
    """Compare baseline vs SSL approaches."""
    print("="*80)
    print("COMPARING BASELINE VS SSL APPROACHES")
    print("="*80)
    
    # Run baseline
    baseline_results = run_baseline_experiment()
    
    print("\n" + "="*80)
    
    # Run SSL
    ssl_results = run_full_ssl_experiment()
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    baseline_acc = baseline_results['test_accuracy']
    ssl_acc = ssl_results['test_accuracy']
    improvement = ssl_acc - baseline_acc
    
    print(f"Baseline Test Accuracy: {baseline_acc:.2f}%")
    print(f"SSL Test Accuracy: {ssl_acc:.2f}%")
    print(f"Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("✅ SSL approach shows improvement!")
    else:
        print("❌ SSL approach did not improve over baseline")


def main():
    """Main function demonstrating different experiment configurations."""
    print("SSL Experiment Framework")
    print("Choose an experiment to run:")
    print("1. Small experiment (quick test)")
    print("2. Full SSL experiment")
    print("3. Baseline experiment (no pre-training)")
    print("4. Compare baseline vs SSL")
    print("5. Custom experiment")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        results = run_small_experiment()
    elif choice == "2":
        results = run_full_ssl_experiment()
    elif choice == "3":
        results = run_baseline_experiment()
    elif choice == "4":
        compare_experiments()
        return
    elif choice == "5":
        # Custom experiment example
        config = ExperimentConfig()
        
        # Customize configuration
        config.experiment_name = "custom_ssl_experiment"
        config.data.labeled_size = 2000
        config.data.batch_size = 64
        config.training.autoencoder_epochs = 50
        config.training.classifier_epochs = 30
        config.model.dropout_rate = 0.3
        
        results = run_ssl_experiment(config)
    else:
        print("Invalid choice")
        return
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
