from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import numpy as np
import time
import os
import copy
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import random
from azureml.core.run import Run
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import wandb
from dotenv import load_dotenv
import pandas as pd
import datetime
import csv
from pathlib import Path

# get the Azure ML run object
run = Run.get_context()

class CustomImageDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_folder_dataset)

    def __getitem__(self, idx):
        path, target = self.image_folder_dataset.imgs[idx]
        image = Image.open(path)
        
        if self.transform:
            image = self.transform(image)

        return image, target, path  # image, label, and file path

def custom_collate(batch):
    # Separate image and label tensors from the file paths
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    paths = [item[2] for item in batch]
    
    # Return the collated data
    return images, labels, paths

def set_seed(seed=42):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior

def load_data(data_dir, seed=42, get_weights_only=False, use_weighted_loss=True):
    """Load the train/val data.
    
    Args:
        data_dir: Directory containing the data
        seed: Random seed for reproducibility
        get_weights_only: If True, only return class weights without loading data
        use_weighted_loss: If True, use weighted loss. If False, use weighted sampling.
    
    Returns:
        dataloaders, dataset_sizes, class_names, class_weights_tensor
    """

    # Set seed before applying transformations for reproducibility
    set_seed(seed)

    # Initialize your datasets 
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
    
    # Print class distribution and calculate class weights
    train_dataset = image_datasets['train']
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.imgs:
        class_counts[label] += 1
    
    print("\nOriginal class distribution in training:")
    for i, count in enumerate(class_counts):
        print(f"Class {train_dataset.classes[i]}: {count} images")
    
    # Calculate class weights for loss function
    # Inverse frequency weighting
    class_weights = [1.0 / count for count in class_counts]
    
    # Normalize weights to sum to number of classes (optional)
    total_weight = sum(class_weights)
    class_weights = [weight * len(class_weights) / total_weight for weight in class_weights]
    
    print("Class weights:", class_weights)
    
    # Convert to tensor for use in loss function
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    # If only weights are needed, return early
    if get_weights_only:
        return None, None, train_dataset.classes, class_weights_tensor
    
    # Continue with data loading if full dataloaders are needed
    data_transforms = {
        'train':transforms.Compose([
        transforms.CenterCrop((1500, 1024)),  # Center cropping
        transforms.RandomHorizontalFlip(p=0.3),  # Random horizontal flip
        transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.3),  # Random rotation (-30 to +30 degrees)
        transforms.RandomResizedCrop((1500, 1024), scale=(0.8, 1.2)),  # Random scaling
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
        transforms.CenterCrop((1500,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    custom_datasets = {x: CustomImageDataset(image_datasets[x], transform=data_transforms[x]) for x in ['train', 'val']}

    if use_weighted_loss:
        # Use weighted loss function with regular shuffling
        print("Using weighted loss approach (class-weighted CrossEntropyLoss)")
        dataloaders = {
            'train': DataLoader(
                custom_datasets['train'],
                batch_size=16,
                shuffle=True,
                num_workers=1,
                collate_fn=custom_collate
            ),
            'val': DataLoader(
                custom_datasets['val'],
                batch_size=16,
                shuffle=True,
                num_workers=1,
                collate_fn=custom_collate
            )
        }
    else:
        # Use weighted sampling approach
        print("Using weighted sampling approach (WeightedRandomSampler)")
        
        # Calculate weights for balanced sampling
        weights = [1.0 / class_counts[label] for _, label in train_dataset.imgs]
        
        # Create a generator with fixed seed for reproducible sampling
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator
        )
        
        dataloaders = {
            'train': DataLoader(
                custom_datasets['train'],
                batch_size=16,
                sampler=sampler,
                num_workers=1,
                collate_fn=custom_collate
            ),
            'val': DataLoader(
                custom_datasets['val'],
                batch_size=16,
                shuffle=True,
                num_workers=1,
                collate_fn=custom_collate
            )
        }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names, class_weights_tensor

def load_test_data(data_dir):
    """Load the test data."""

    # Data augmentation and normalization for training
    data_transforms = {
        'test': transforms.Compose([
        transforms.CenterCrop((1500,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['test']}
    custom_datasets = {x: CustomImageDataset(image_datasets[x], transform=data_transforms[x]) for x in ['test']}

    # Use the custom_collate function in DataLoader
    dataloaders = {
        x: DataLoader(custom_datasets[x], batch_size=32, shuffle=True, num_workers=1, collate_fn=custom_collate)
        for x in ['test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    class_names = image_datasets['test'].classes

    return dataloaders, dataset_sizes, class_names


def train_model(model, criterion, optimizer, scheduler, config):
    """Train the model with parameters from config."""
    data_dir = config['input_data']
    num_epochs = config['num_epochs']
    patience = config['patience']
    use_weighted_loss = config.get('use_weighted_loss', True)

    # Load training/validation data
    dataloaders, dataset_sizes, class_names, _ = load_data(data_dir, use_weighted_loss=use_weighted_loss)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Move class weights to the same device as model if they exist
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0  # Counter for tracking epochs with no improvement in validation accuracy

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels, filename in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            ## Scheduler step
            if phase == 'train':
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau requires validation loss (not available yet in train phase)
                    pass 
                else:
                    scheduler.step()
            elif phase == 'val' and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # Log the epoch loss and accuracy
            run.log(name=f'{phase}_loss', value=float(epoch_loss))
            run.log(name=f'{phase}_acc', value=float(epoch_acc))


            # Deep copy the model if we get a better validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0  # Reset counter after improvement
                # log the best val accuracy to AML run
                run.log('best_val_acc', float(best_acc))
            elif phase == 'val':
                epochs_no_improve += 1  # Increment counter if no improvement

        # Check early stopping condition
        if epochs_no_improve == patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def fine_tune_model(config):
    """Load a pretrained model and reset the final fully connected layer using config."""
    num_epochs = config['num_epochs']
    data_dir = config['input_data']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    patience = config['patience']
    p_dropout = config.get('p_dropout', 0.3)
    step_size = config.get('step_size', 7)
    gamma = config.get('gamma', 0.1)
    model_type = config.get('model_type', 'resnet18')
    use_weighted_loss = config.get('use_weighted_loss', True)

    # log the hyperparameter metrics to the AML run
    run.log('lr', float(learning_rate))
    run.log('momentum', float(momentum))
    run.log('p_dropout', float(p_dropout))
    run.log('step_size', float(step_size))
    run.log('gamma', float(gamma))
    run.log('model_type', model_type)
    run.log('use_weighted_loss', use_weighted_loss)

    # Get class weights for loss function without loading full dataloaders
    _, _, _, class_weights = load_data(data_dir, get_weights_only=True, use_weighted_loss=use_weighted_loss)
    
    # Log class weights
    for i, weight in enumerate(class_weights):
        run.log(f'class_{i}_weight', float(weight))

    # Model selection based on model_type parameter
    if model_type.lower() == 'resnet18':
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(num_ftrs, 2)  # 2 classes: male/female
        )
    elif model_type.lower() == 'resnet34':
        model_ft = models.resnet34(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(num_ftrs, 2)
        )
    elif model_type.lower() == 'densenet121':
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(num_ftrs, 2)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from: resnet18, resnet34, densenet121")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_ft.to(device)

    # Create loss criterion based on approach
    if use_weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted CrossEntropyLoss with class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss with WeightedRandomSampler")

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)

    # Initialize scheduler based on config
    scheduler_type = config.get('scheduler_type', 'step')
    if scheduler_type.lower() == 'plateau':
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer_ft, 
            mode='min', 
            patience=config.get('scheduler_patience', 2), 
            factor=config.get('scheduler_factor', 0.5), 
            verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler (patience={config.get('scheduler_patience', 2)}, factor={config.get('scheduler_factor', 0.5)})")
    else:
        # Default: StepLR
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=step_size, gamma=gamma)
        print(f"Using StepLR scheduler (step_size={step_size}, gamma={gamma})")

    model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, config)

    return model


def test_model(model, data_dir, output_path):
    """Test the model on the test set and calculate metrics."""
    model.eval()  # Set model to evaluation mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    true_labels = []
    preds = []
    misclassified_info = []

    dataloaders, dataset_sizes, class_names = load_test_data(data_dir)
    
    # Print class names and their indices
    print("\nTesting with classes:", class_names)

    with torch.no_grad():
        for inputs, labels, filenames in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)  # Ensure labels are also moved to the correct device
            
            
            # Forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            # Save predictions and true labels
            preds.extend(predictions.view(-1).tolist())
            true_labels.extend(labels.view(-1).tolist())

            # Check for misclassifications and record them
            for filename, label, prediction in zip(filenames, labels, predictions):
                if label != prediction:
                    misclassified_info.append((filename, class_names[label], class_names[prediction]))

    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, zero_division=1)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)

    # Calculate per-class metrics
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        class_precision = precision_score(true_labels, preds, labels=[i], average='binary', zero_division=1)
        class_recall = recall_score(true_labels, preds, labels=[i], average='binary')
        class_f1 = f1_score(true_labels, preds, labels=[i], average='binary')
        print(f"\nMetrics for class '{class_name}':")
        print(f"Precision: {class_precision:.4f}")
        print(f"Recall: {class_recall:.4f}")
        print(f"F1-Score: {class_f1:.4f}")
        
        # Log per-class metrics
        run.log(f'{class_name}_precision', round(float(class_precision), 4))
        run.log(f'{class_name}_recall', round(float(class_recall), 4))
        run.log(f'{class_name}_f1', round(float(class_f1), 4))

    # Generate the confusion matrix and display
    unique_classes = np.unique(np.concatenate((true_labels, preds)))
    cm = confusion_matrix(true_labels, preds, labels=unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
    disp.plot()
    plt.show()
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png')) # Save the confusion matrix to a file
    
    # Extract TN, FP, FN, TP for binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Negative Predictive Value
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Specificity
    else:
        npv = specificity = 0  # Ensure backward compatibility

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (PPV): {precision:.4f}')
    print(f'Negative Predictive Value (NPV): {npv:.4f}')
    print(f'Recall (Sensitivity): {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Log metrics to Azure ML run with same names and precision
    run.log('Accuracy', round(float(accuracy), 4))
    run.log('Precision (PPV)', round(float(precision), 4))
    run.log('Negative Predictive Value (NPV)', round(float(npv), 4))
    run.log('Recall (Sensitivity)', round(float(recall), 4))
    run.log('Specificity', round(float(specificity), 4))
    run.log('F1 Score', round(float(f1), 4))

    # Save misclassified cases along with metrics
    with open(os.path.join(output_path, 'results_summary.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (PPV): {precision:.4f}\n")
        f.write(f"Negative Predictive Value (NPV): {npv:.4f}\n")
        f.write(f"Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Misclassified cases:\n")
        for filename, true_label, predicted_label in misclassified_info:
            f.write(f"Filename: {filename}, True Label: {true_label}, Predicted: {predicted_label}\n")
    
    return accuracy, precision, npv, recall, specificity, f1, cm


# Convenience functions for hyperparameter sweep
def create_summary_file(output_dir):
    """Create a master summary file to track all runs."""
    summary_path = os.path.join(output_dir, "sweep_summary.csv")
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'run_id', 'timestamp', 'view', 'model_type', 'learning_rate', 'momentum', 'p_dropout',
                'step_size', 'gamma', 'use_weighted_loss',
                'accuracy', 'precision', 'npv', 'recall', 'specificity', 'f1',
                'best_val_acc', 'epochs_completed', 'early_stopped', 'run_time_mins'
            ])
    
    return summary_path

def update_summary_file(summary_path, run_id, config, metrics, start_time, epochs_completed, early_stopped):
    """Add a new run to the summary file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_time_mins = (time.time() - start_time) / 60
    
    row = [
        run_id, timestamp,
        config.get('view', 'N/A'),
        config.get('model_type', 'N/A'),
        config.get('learning_rate', 'N/A'), 
        config.get('momentum', 'N/A'),
        config.get('p_dropout', 'N/A'),
        config.get('step_size', 'N/A'),
        config.get('gamma', 'N/A'),
        config.get('use_weighted_loss', 'N/A'),
        metrics['accuracy'], metrics['precision'], metrics['npv'],
        metrics['recall'], metrics['specificity'], metrics['f1'],
        metrics.get('best_val_acc', 'N/A'), 
        epochs_completed, early_stopped, f"{run_time_mins:.2f}"
    ]
    
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f"Updated summary file with run {run_id}")
    
    # Also create a sorted view
    try:
        df = pd.read_csv(summary_path)
        sorted_df = df.sort_values(by=['accuracy'], ascending=False)
        sorted_path = os.path.join(os.path.dirname(summary_path), "sweep_summary_sorted.csv")
        sorted_df.to_csv(sorted_path, index=False)
        
        # Create a mini version with just the top 5 runs
        top_runs = sorted_df.head(5)
        top_path = os.path.join(os.path.dirname(summary_path), "sweep_top_5_runs.csv")
        top_runs.to_csv(top_path, index=False)
    except Exception as e:
        print(f"Warning: Could not create sorted summary: {e}")
        
def plot_summary_results(summary_path, output_dir):
    """Create visualization of hyperparameter sweep results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = pd.read_csv(summary_path)
        if len(df) < 2:
            print("Not enough data points for plotting")
            return
            
        # Create plot directory
        plot_dir = os.path.join(output_dir, "sweep_plots")
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        # Plot accuracy vs dropout rate
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='p_dropout', y='accuracy', hue='learning_rate', size='momentum', data=df)
        plt.title('Accuracy vs Dropout Rate')
        plt.savefig(os.path.join(plot_dir, 'accuracy_vs_dropout.png'))
        plt.close()
        
        # Plot learning rate vs accuracy
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='learning_rate', y='accuracy', hue='p_dropout', size='momentum', data=df)
        plt.xscale('log')
        plt.title('Accuracy vs Learning Rate (log scale)')
        plt.savefig(os.path.join(plot_dir, 'accuracy_vs_lr.png'))
        plt.close()
        
        # Create a pairplot of all parameters if we have enough data points
        if len(df) >= 3:
            plt.figure(figsize=(12, 10))
            params = ['learning_rate', 'momentum', 'p_dropout', 'accuracy', 'f1']
            sns.pairplot(df[params], diag_kind='kde', height=2)
            plt.savefig(os.path.join(plot_dir, 'parameter_pairplot.png'))
            plt.close()
            
    except Exception as e:
        print(f"Warning: Could not create summary plots: {e}")
        print("Please install matplotlib and seaborn: pip install matplotlib seaborn")

        
def load_hyperparams(config_path):
    """Load and flatten the nested hyperparams JSON."""
    import json
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    # Flatten the nested structure for compatibility
    flat_config = {}
    for section in data.values():
        flat_config.update(section)
    return flat_config

def main():
    print("Torch version:", torch.__version__)

    # get command-line arguments
    parser = argparse.ArgumentParser(description='Train gender classification model for forensic sex identification')
    parser.add_argument('--config', type=str, default='hyperparams.json', help='Path to hyperparams JSON file')
    parser.add_argument('--input_data', type=str, help='Path to the input data directory')
    parser.add_argument('--view', type=str, choices=['ap', 'lateral'], help='Radiograph view type')
    parser.add_argument('--model_type', type=str, choices=['resnet18', 'resnet34', 'densenet121'], help='Model architecture')
    # ... keep other arguments as they are for overrides ...

    parser.add_argument('--num_epochs', type=int, default=25,
                        help='number of epochs to train')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='output directory')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--p_dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--step_size', type=int, default=7, help='step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma for StepLR scheduler')
    parser.add_argument('--scheduler_type', type=str, default='step', choices=['step', 'plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler_patience', type=int, default=2, help='patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='factor for ReduceLROnPlateau')
    parser.add_argument('--use_weighted_loss', action='store_true', 
                        help='use weighted loss for class imbalance (default: False)')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for hyperparameter tuning')
    parser.add_argument('--wandb_project', type=str, default='forensic-sex-classification', help='wandb project name')
    parser.add_argument('--wandb_sweep', action='store_true', help='create and run wandb sweep')
    parser.add_argument('--wandb_sweep_count', type=int, default=30, help='number of wandb sweep runs')
    parser.add_argument('--wandb_api_key', type=str, default=None, help='wandb API key')
    args = parser.parse_args()

    # Base config from JSON if exists
    config = {}
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        config = load_hyperparams(args.config)
    
    # Override with command line arguments if provided
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        # Only override if the argument was explicitly provided (not None) or is a flag
        if value is not None and key != 'config':
            config[key] = value

    # Validate required parameters
    if 'input_data' not in config or 'view' not in config:
        parser.error("--input_data and --view are required (either in config or as CLI args)")

    data_dir = config['input_data']
    print(f"Data directory: {data_dir}")
    print(f"View: {config['view']}")
    print(f"Model: {config.get('model_type', 'resnet18')}")
    
    # Ensure output directory exists
    output_dir = config.get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary file
    summary_path = create_summary_file(output_dir)
    
    # Log the view type to Azure ML
    run.log('view', config['view'])
    
    if config.get('use_wandb', False):
        try:
            load_dotenv()  # Load variables from .env file
            env_api_key = os.environ.get('WANDB_KEY')
            if env_api_key:
                print("Using WANDB_API_KEY from .env file for authentication")
                wandb.login(key=env_api_key)
            elif args.wandb_api_key:
                print("Using provided wandb API key for authentication")
                wandb.login(key=args.wandb_api_key)
            else:
                print("No wandb API key found. Attempting default wandb authentication...")
                wandb.login()
            
            def train_with_wandb_wrapper():
                # This wrapper function maintains compatibility with existing code
                with wandb.init() as wb_run:
                    # Access hyperparameters from wandb
                    config = dict(wandb.config)
                    
                    # Ensure non-sweep parameters are also in config
                    config['input_data'] = args.input_data
                    config['num_epochs'] = args.num_epochs
                    config['patience'] = args.patience
                    config['output_dir'] = args.output_dir
                    config['view'] = args.view
                    config['model_type'] = args.model_type
                    config['use_weighted_loss'] = args.use_weighted_loss
                    
                    # Create output directory per run
                    run_output_dir = os.path.join(args.output_dir, f"wandb_{wb_run.id}")
                    os.makedirs(run_output_dir, exist_ok=True)
                    
                    # Track training start time
                    start_time = time.time()
                    
                    # Train the model with wandb hyperparameters
                    try:
                        # Train the model
                        model = fine_tune_model(config)
                        
                        # Test the model and report metrics
                        accuracy, precision, npv, recall, specificity, f1, cm = test_model(
                            model, config['input_data'], output_path=run_output_dir
                        )
                        
                        # Create metrics dict for summary file
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'npv': npv,
                            'recall': recall,
                            'specificity': specificity,
                            'f1': f1,
                            'best_val_acc': wb_run.summary.get('best_val_acc', 0)
                        }
                        
                        # Update the summary file
                        config_dict = {k: v for k, v in config.items()}
                        config_dict['view'] = args.view
                        config_dict['model_type'] = args.model_type
                        config_dict['use_weighted_loss'] = args.use_weighted_loss
                        
                        update_summary_file(
                            summary_path, 
                            wb_run.id, 
                            config_dict, 
                            metrics,
                            start_time,
                            args.num_epochs,
                            False
                        )
                        
                        # Save the model
                        torch.save(model, os.path.join(run_output_dir, 'model.pt'))
                        
                        # Save confusion matrix with hyperparameters in the filename
                        cm_filename = f"cm_{args.view}_lr{config.learning_rate:.6f}_m{config.momentum:.2f}_d{config.p_dropout:.2f}.png"
                        plt.figure(figsize=(8, 6))
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                        disp.plot()
                        plt.title(f"View={args.view}, LR={config.learning_rate:.6f}, M={config.momentum:.2f}, D={config.p_dropout:.2f}\nAcc={accuracy:.4f}, F1={f1:.4f}")
                        plt.savefig(os.path.join(args.output_dir, cm_filename))
                        plt.close()
                        
                    except Exception as e:
                        import traceback
                        print(f"Error in run {wb_run.id}: {e}")
                        traceback.print_exc()
            
            if args.wandb_sweep:
                # Define hyperparameter search space
                sweep_config = {
                    'method': 'bayes',
                    'metric': {
                        'name': 'best_val_acc',
                        'goal': 'maximize'
                    },
                    'parameters': {
                        'learning_rate': {
                            'distribution': 'log_uniform_values',
                            'min': 0.0001,
                            'max': 0.01
                        },
                        'momentum': {
                            'distribution': 'uniform',
                            'min': 0.6,
                            'max': 0.99
                        },
                        'p_dropout': {
                            'distribution': 'uniform',
                            'min': 0.0,
                            'max': 0.7
                        },
                        'step_size': {
                            'distribution': 'int_uniform',
                            'min': 2,
                            'max': 8
                        },
                        'gamma': {
                            'distribution': 'uniform',
                            'min': 0.1,
                            'max': 0.7
                        }
                    }
                }
                
                # Initialize sweep
                sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
                
                # Run the sweep
                wandb.agent(sweep_id, function=train_with_wandb_wrapper, count=args.wandb_sweep_count)
                
                # Generate final summary
                if os.path.exists(summary_path):
                    print("\n==== Hyperparameter Sweep Summary ====")
                    df = pd.read_csv(summary_path)
                    print(f"Total runs completed: {len(df)}")
                    
                    if len(df) > 0:
                        df_sorted = df.sort_values(by=['accuracy'], ascending=False)
                        print("\nTop 5 performing configurations:")
                        for i, row in df_sorted.head(5).iterrows():
                            print(f"Run {i+1}: Accuracy={row['accuracy']:.4f}, F1={row['f1']:.4f}")
                            print(f"    View={row['view']}, Model={row['model_type']}")
                            print(f"    LR={row['learning_rate']}, Momentum={row['momentum']}, Dropout={row['p_dropout']}")
                        
                        plot_summary_results(summary_path, args.output_dir)
                
            else:
                # Single wandb run
                wandb.init(
                    project=args.wandb_project,
                    config={
                        "learning_rate": args.learning_rate,
                        "momentum": args.momentum,
                        "p_dropout": args.p_dropout,
                        "step_size": args.step_size,
                        "gamma": args.gamma,
                        "num_epochs": args.num_epochs,
                        "patience": args.patience
                    }
                )
                train_with_wandb_wrapper()
                
        except ImportError:
            print("wandb not installed. Please install with 'pip install wandb'")
            print("Falling back to normal training")
            args.use_wandb = False

    if not args.use_wandb:
        # Normal training without wandb
        start_time = time.time()
        
        # Create config from args
        config = vars(args)
        
        model = fine_tune_model(config)
        
        accuracy, precision, npv, recall, specificity, f1, cm = test_model(model, args.input_data, output_path=args.output_dir)
        torch.save(model, os.path.join(args.output_dir, 'model.pt'))
        
        # Update summary file
        try:
            run_id = str(run.id) if hasattr(run, 'id') else f"local_{int(time.time())}"
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'npv': npv,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'best_val_acc': getattr(run, 'best_val_acc', 0)
            }
            
            config = {
                'view': args.view,
                'model_type': args.model_type,
                'learning_rate': args.learning_rate,
                'momentum': args.momentum,
                'p_dropout': args.p_dropout,
                'step_size': args.step_size,
                'gamma': args.gamma,
                'use_weighted_loss': args.use_weighted_loss
            }
            
            update_summary_file(
                summary_path,
                run_id,
                config,
                metrics,
                start_time,
                args.num_epochs,
                False
            )
            
            plot_summary_results(summary_path, args.output_dir)
        except Exception as e:
            print(f"Warning: Could not update summary file: {e}")
    
    print("\n==== Training Complete ====")
    print(f"View: {args.view}")
    print(f"Model: {args.model_type}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

