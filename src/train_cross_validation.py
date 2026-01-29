from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

import numpy as np
import time
import os
import copy
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import random
from azureml.core.run import Run
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
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

def create_cv_folds(dataset, n_splits=8, seed=42):
    """
    Create cross-validation folds with stratification to ensure balanced class distribution.
    Returns indices for each fold.
    """
    # Extract labels
    labels = [label for _, label in dataset.imgs]
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Generate fold indices
    fold_indices = []
    for _, val_idx in skf.split(np.zeros(len(labels)), labels):
        fold_indices.append(val_idx)
        
    return fold_indices

def load_data_for_cv(data_dir, fold_idx=None, use_original_val=False, seed=42, use_weighted_loss=True):
    """
    Load data for cross-validation.
    
    Args:
        data_dir: Directory containing the data
        fold_idx: Index of the fold to use as validation (0-7)
        use_original_val: Whether to use the original validation set
        seed: Random seed for reproducibility
        use_weighted_loss: Whether to use weighted loss function instead of weighted sampling
    
    Returns:
        dataloaders, dataset_sizes, class_names, class_weights_tensor (if use_weighted_loss=True)
        or dataloaders, dataset_sizes, class_names (if use_weighted_loss=False)
    """
    # Set seed for reproducibility
    set_seed(seed)

    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop((1500, 1024)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.3),
            transforms.RandomResizedCrop((1500, 1024), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop((1500, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load original datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
    
    # Special case: Use original validation set (Run 1)
    if use_original_val:
        print("Using original validation set for CV fold")
        custom_datasets = {x: CustomImageDataset(image_datasets[x], transform=data_transforms[x]) for x in ['train', 'val']}
        
        # Calculate class statistics for training dataset
        train_dataset = image_datasets['train']
        class_counts = [0] * len(train_dataset.classes)
        for _, label in train_dataset.imgs: 
            class_counts[label] += 1
        
        print("\nClass distribution in training:")
        for i, count in enumerate(class_counts):
            print(f"Class {train_dataset.classes[i]}: {count} images")
        
        if use_weighted_loss:
            # Calculate class weights for loss function (inverse frequency)
            class_weights = [1.0 / count for count in class_counts]
            
            # Normalize weights to sum to number of classes
            total_weight = sum(class_weights)
            class_weights = [weight * len(class_weights) / total_weight for weight in class_weights]
            
            print("Class weights for loss function:", class_weights)
            
            # Convert to tensor for use in loss function
            class_weights_tensor = torch.FloatTensor(class_weights)
            
            # Use regular DataLoader without weighted sampling
            dataloaders = {
                'train': DataLoader(
                    custom_datasets['train'],
                    batch_size=16,
                    shuffle=True,  # Use shuffling instead of weighted sampling
                    num_workers=1,
                    collate_fn=custom_collate
                ),
                'val': DataLoader(
                    custom_datasets['val'],
                    batch_size=16,
                    shuffle=True,  # Enabled shuffling for validation
                    num_workers=1,
                    collate_fn=custom_collate
                )
            }
        else:
            # Original approach: Calculate weights for balanced sampling
            weights = [1.0 / class_counts[label] for _, label in train_dataset.imgs]
            
            # Create generator with fixed seed
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
                    shuffle=True,  # Enabled shuffling for validation
                    num_workers=1,
                    collate_fn=custom_collate
                )
            }
            
            # Create a dummy tensor for compatibility with the new function signature
            class_weights_tensor = None
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        
    else:
        # For runs 2-9: Create custom train/val split from training data
        # and use original validation set as part of training
        if fold_idx is None or fold_idx < 0 or fold_idx >= 8:
            raise ValueError("fold_idx must be between 0 and 7 for custom CV splits")
        
        # Create folds from training data
        train_dataset = image_datasets['train']
        val_dataset = image_datasets['val']
        
        fold_indices = create_cv_folds(train_dataset, n_splits=8, seed=seed)
        
        # Select current fold as validation
        val_indices = fold_indices[fold_idx]
        
        # All other folds are for training
        train_indices = []
        for i in range(8):
            if i != fold_idx:
                train_indices.extend(fold_indices[i])
        
        print(f"\nUsing fold {fold_idx+1}/8 as validation")
        print(f"Original training samples used for training: {len(train_indices)}")
        print(f"Original validation samples added to training: {len(val_dataset)}")
        print(f"Validation samples (from original training): {len(val_indices)}")
        
        # Calculate class counts for reporting
        train_labels = [train_dataset.imgs[i][1] for i in train_indices]
        val_labels = [train_dataset.imgs[i][1] for i in val_indices]
        
        class_counts_train = [0] * len(train_dataset.classes)
        class_counts_val = [0] * len(train_dataset.classes)
        class_counts_original_val = [0] * len(train_dataset.classes)
        
        for label in train_labels:
            class_counts_train[label] += 1
        for label in val_labels:
            class_counts_val[label] += 1
        for _, label in val_dataset.imgs:
            class_counts_original_val[label] += 1
        
        print("\nClass distribution in training fold:")
        for i, count in enumerate(class_counts_train):
            print(f"Class {train_dataset.classes[i]}: {count} images")
        
        print("\nClass distribution in original validation set (added to training):")
        for i, count in enumerate(class_counts_original_val):
            print(f"Class {train_dataset.classes[i]}: {count} images")
            
        print("\nClass distribution in validation fold:")
        for i, count in enumerate(class_counts_val):
            print(f"Class {train_dataset.classes[i]}: {count} images")
        
        # Create transformations for both datasets
        train_custom_orig = CustomImageDataset(train_dataset, transform=data_transforms['train'])
        val_as_train_custom = CustomImageDataset(val_dataset, transform=data_transforms['train'])
        val_custom = CustomImageDataset(train_dataset, transform=data_transforms['val'])
        
        # Create a subset for validation (from original training)
        val_custom_subset = Subset(val_custom, val_indices)
        
        # Create a subset for part of training (from original training)
        train_custom_subset = Subset(train_custom_orig, train_indices)
        
        # Calculate combined class counts
        combined_train_labels = train_labels.copy()  # Labels from 7 folds
        combined_train_labels.extend([label for _, label in val_dataset.imgs])  # Add labels from original val set
        
        # Calculate class counts for the combined dataset
        combined_class_counts = [0] * len(train_dataset.classes)
        for label in combined_train_labels:
            combined_class_counts[label] += 1
            
        print("\nClass distribution in combined training set (7 folds + original val):")
        for i, count in enumerate(combined_class_counts):
            print(f"Class {train_dataset.classes[i]}: {count} images")
        
        # We need to create a custom dataset that combines the two sources
        class CombinedDataset(Dataset):
            def __init__(self, dataset1, dataset2):
                self.dataset1 = dataset1
                self.dataset2 = dataset2
                self.len1 = len(dataset1)
                self.len2 = len(dataset2)
            
            def __len__(self):
                return self.len1 + self.len2
            
            def __getitem__(self, idx):
                if idx < self.len1:
                    return self.dataset1[idx]
                else:
                    return self.dataset2[idx - self.len1]
        
        # Create the combined dataset
        combined_train_dataset = CombinedDataset(train_custom_subset, val_as_train_custom)
        
        if use_weighted_loss:
            # Calculate class weights for loss function
            class_weights = [1.0 / count for count in combined_class_counts]
            
            # Normalize weights to sum to number of classes
            total_weight = sum(class_weights)
            class_weights = [weight * len(class_weights) / total_weight for weight in class_weights]
            
            print("Class weights for loss function:", class_weights)
            
            # Convert to tensor for use in loss function
            class_weights_tensor = torch.FloatTensor(class_weights)
            
            # Create dataloaders without weighted sampling
            dataloaders = {
                'train': DataLoader(
                    combined_train_dataset,
                    batch_size=16,
                    shuffle=True,  # Use shuffling instead of weighted sampling
                    num_workers=1,
                    collate_fn=custom_collate
                ),
                'val': DataLoader(
                    val_custom_subset,
                    batch_size=16,
                    shuffle=True,  # Enabled shuffling for validation
                    num_workers=1,
                    collate_fn=custom_collate
                )
            }
        else:
            # Original approach: Calculate weights for balanced sampling
            combined_weights = [1.0 / combined_class_counts[label] for label in combined_train_labels]
            
            # Create generator with fixed seed
            generator = torch.Generator()
            generator.manual_seed(seed)
            
            # Create sampler for the combined dataset
            sampler = WeightedRandomSampler(
                weights=combined_weights,
                num_samples=len(combined_train_labels),
                replacement=True,
                generator=generator
            )
            
            # Create dataloaders
            dataloaders = {
                'train': DataLoader(
                    combined_train_dataset,
                    batch_size=16,
                    sampler=sampler,
                    num_workers=1,
                    collate_fn=custom_collate
                ),
                'val': DataLoader(
                    val_custom_subset,
                    batch_size=16,
                    shuffle=True,  # Enabled shuffling for validation
                    num_workers=1,
                    collate_fn=custom_collate
                )
            }
            
            # Create a dummy tensor for compatibility with the new function signature
            class_weights_tensor = None
        
        dataset_sizes = {
            'train': len(combined_train_dataset),
            'val': len(val_indices)
        }
        class_names = train_dataset.classes
    
    if use_weighted_loss:
        return dataloaders, dataset_sizes, class_names, class_weights_tensor
    else:
        return dataloaders, dataset_sizes, class_names, None

def train_model_cv(model, criterion_or_weights, optimizer, scheduler, dataloaders, dataset_sizes, config):
    """Train model for a single cross-validation fold using config."""
    num_epochs = config['num_epochs']
    patience = config['patience']
    use_weighted_loss = config.get('use_weighted_loss', True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up criterion based on whether we're using weighted loss
    if use_weighted_loss:
        # Using weighted loss - criterion_or_weights is the class weights tensor
        class_weights = criterion_or_weights.to(device) if criterion_or_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Using traditional approach - criterion_or_weights is already the criterion
        criterion = criterion_or_weights

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels, paths in dataloaders[phase]:
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

            if phase == 'train':
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    pass
                else:
                    scheduler.step()
            elif phase == 'val' and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Log metrics to AML run
            run.log(name=f'{phase}_loss', value=float(epoch_loss))
            run.log(name=f'{phase}_acc', value=float(epoch_acc))

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                run.log('best_val_acc', float(best_acc))
            elif phase == 'val':
                epochs_no_improve += 1

        # Check early stopping
        if epochs_no_improve == patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Now perform a final evaluation pass with the best model to collect metrics
    model.eval()
    val_metrics = {
        'preds': [],
        'true': [],
        'paths': []
    }
    
    print("Performing final validation evaluation for metrics...")
    with torch.no_grad():
        for inputs, labels, paths in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            val_metrics['preds'].extend(preds.cpu().numpy())
            val_metrics['true'].extend(labels.cpu().numpy())
            val_metrics['paths'].extend(paths)
    
    # Calculate final validation metrics
    metrics = calculate_metrics(val_metrics['true'], val_metrics['preds'])
    metrics['best_val_acc'] = float(best_acc)
    
    # Print sample counts to verify correct validation set size
    print(f"Validation samples evaluated: {len(val_metrics['true'])}")
    
    return model, metrics

def calculate_metrics(y_true, y_pred):
    """Calculate all required metrics for validation data"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract TN, FP, FN, TP for binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Negative Predictive Value
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Specificity
    else:
        npv = specificity = 0  # Ensure backward compatibility
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'npv': npv,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': cm
    }

def run_cross_validation(config):
    """Run 9-fold cross-validation using config dictionary."""
    num_epochs = config['num_epochs']
    p_dropout = config.get('p_dropout', 0.3)
    data_dir = config['input_data']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    patience = config['patience']
    step_size = config.get('step_size', 7)
    gamma = config.get('gamma', 0.1)
    model_type = config.get('model_type', 'resnet18')
    use_weighted_loss = config.get('use_weighted_loss', True)

    # Log hyperparameters to AML run
    run.log('lr', float(learning_rate))
    run.log('momentum', float(momentum))
    run.log('p_dropout', float(p_dropout))
    run.log('step_size', int(step_size))
    run.log('gamma', float(gamma))
    run.log('model_type', model_type)
    run.log('use_weighted_loss', use_weighted_loss)
    
    # Create directory for CV results
    cv_results_dir = os.path.join(config.get('output_dir', './outputs'), 'cv_results')
    os.makedirs(cv_results_dir, exist_ok=True)
    
    # Initialize results tracking
    all_metrics = []
    all_models = []
    
    # Prepare for 9-fold CV
    print("\n======= Starting 9-fold Cross-Validation =======")
    
    # Run 1: Use original validation set
    print("\n===== Fold 1: Using original validation set =====")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Run 1: Use the original validation set
    dataloaders, dataset_sizes, class_names, class_weights = load_data_for_cv(
        data_dir, 
        use_original_val=True, 
        use_weighted_loss=use_weighted_loss
    )
    
    # Initialize model for fold 1
    if model_type.lower() == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(num_ftrs, 2)
        )
    elif model_type.lower() == 'resnet34':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(num_ftrs, 2)
        )
    elif model_type.lower() == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(num_ftrs, 2)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)
    
    # Define criterion, optimizer and scheduler based on approach
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # Initialize scheduler based on config
    scheduler_type = config.get('scheduler_type', 'step')
    if scheduler_type.lower() == 'plateau':
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=config.get('scheduler_patience', 2), 
            factor=config.get('scheduler_factor', 0.5), 
            verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler (patience={config.get('scheduler_patience', 2)}, factor={config.get('scheduler_factor', 0.5)})")
    else:
        # Default: StepLR
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"Using StepLR scheduler (step_size={step_size}, gamma={gamma})")
    
    if not use_weighted_loss:
        # Using traditional approach with regular criterion
        criterion = nn.CrossEntropyLoss()
        
        # Train and validate for fold 1
        model, metrics = train_model_cv(
            model, criterion, optimizer, exp_lr_scheduler, 
            dataloaders, dataset_sizes, config
        )
    else:
        # Using weighted loss approach with class weights
        model, metrics = train_model_cv(
            model, class_weights, optimizer, exp_lr_scheduler, 
            dataloaders, dataset_sizes, config
        )
    
    # Save fold results
    fold_dir = os.path.join(cv_results_dir, 'fold_1')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Save metrics
    metrics['fold'] = 1
    all_metrics.append(metrics)
    all_models.append(model)
    
    # Save fold model
    torch.save(model, os.path.join(fold_dir, 'model.pt'))
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title(f"Fold 1 - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Runs 2-9: Use remaining combinations of train/val
    for fold_idx in range(8):
        fold_num = fold_idx + 2  # Fold numbers 2-9
        print(f"\n===== Fold {fold_num}: Using custom train/val split =====")
        
        # Load data for this fold
        dataloaders, dataset_sizes, class_names, class_weights = load_data_for_cv(
            data_dir, 
            fold_idx=fold_idx, 
            use_original_val=False, 
            use_weighted_loss=use_weighted_loss
        )
        
        # Initialize a new model for each fold
        if model_type.lower() == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p_dropout),
                nn.Linear(num_ftrs, 2)
            )
        elif model_type.lower() == 'resnet34':
            model = models.resnet34(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p_dropout),
                nn.Linear(num_ftrs, 2)
            )
        elif model_type.lower() == 'densenet121':
            model = models.densenet121(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p_dropout),
                nn.Linear(num_ftrs, 2)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model = model.to(device)
        
        # Define optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        if not use_weighted_loss:
            # Using traditional approach with regular criterion
            criterion = nn.CrossEntropyLoss()
            
            # Train and validate for fold
            model, metrics = train_model_cv(
                model, criterion, optimizer, exp_lr_scheduler, 
                dataloaders, dataset_sizes, config
            )
        else:
            # Using weighted loss approach with class weights
            model, metrics = train_model_cv(
                model, class_weights, optimizer, exp_lr_scheduler, 
                dataloaders, dataset_sizes, config
            )
        
        # Save fold results
        fold_dir = os.path.join(cv_results_dir, f'fold_{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save metrics
        metrics['fold'] = fold_num
        all_metrics.append(metrics)
        all_models.append(model)
        
        # Save fold model
        torch.save(model, os.path.join(fold_dir, 'model.pt'))
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.title(f"Fold {fold_num} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png'))
        plt.close()
    
    # Calculate and report average metrics
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'npv': np.mean([m['npv'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'best_val_acc': np.mean([m['best_val_acc'] for m in all_metrics])
    }
    
    # Also calculate standard deviations
    std_metrics = {
        'accuracy_std': np.std([m['accuracy'] for m in all_metrics]),
        'precision_std': np.std([m['precision'] for m in all_metrics]),
        'npv_std': np.std([m['npv'] for m in all_metrics]),
        'recall_std': np.std([m['recall'] for m in all_metrics]),
        'specificity_std': np.std([m['specificity'] for m in all_metrics]),
        'f1_std': np.std([m['f1'] for m in all_metrics]),
        'best_val_acc_std': np.std([m['best_val_acc'] for m in all_metrics])
    }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_metrics)
    
    # Remove confusion matrix from summary (it's not serializable)
    summary_df = summary_df.drop(columns=['confusion_matrix'])
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(cv_results_dir, 'cv_results.csv'), index=False)
    
    # Print and log final results
    print("\n======= Cross-Validation Results =======")
    print(f"Average Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision_std']:.4f}")
    print(f"Average NPV: {avg_metrics['npv']:.4f} ± {std_metrics['npv_std']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall_std']:.4f}")
    print(f"Average Specificity: {avg_metrics['specificity']:.4f} ± {std_metrics['specificity_std']:.4f}")
    print(f"Average F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
    
    # Log to Azure ML
    for key, value in avg_metrics.items():
        run.log(f'cv_avg_{key}', float(value))
    for key, value in std_metrics.items():
        run.log(f'cv_{key}', float(value))
    
    # Determine best fold based on validation accuracy
    best_fold_idx = np.argmax([m['accuracy'] for m in all_metrics])
    best_fold_num = best_fold_idx + 1
    best_model = all_models[best_fold_idx]
    
    print(f"\nBest performing fold: {best_fold_num} with accuracy {all_metrics[best_fold_idx]['accuracy']:.4f}")
    
    # Save best model
    torch.save(best_model, os.path.join(cv_results_dir, 'best_model.pt'))
    
    return best_model, avg_metrics, all_metrics

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
    parser = argparse.ArgumentParser(description='Train gender classification model with 9-fold cross-validation')
    parser.add_argument('--config', type=str, default='hyperparams.json', help='Path to hyperparams JSON file')
    parser.add_argument('--input_data', type=str, help='Path to the input data directory')
    parser.add_argument('--view', type=str, choices=['ap', 'lateral'], help='Radiograph view type')
    parser.add_argument('--model_type', type=str, choices=['resnet18', 'resnet34', 'densenet121'], help='Model architecture')

    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='output directory')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
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
                        help='use weighted loss instead of weighted sampling (default: False)')
    args = parser.parse_args()

    # Base config from JSON if exists
    config = {}
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        config = load_hyperparams(args.config)
    
    # Override with command line arguments if provided
    arg_dict = vars(args)
    for key, value in arg_dict.items():
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
    
    # Log the view type to Azure ML
    run.log('view', config['view'])
    
    print(f"\nTraining with parameters:")
    for k, v in config.items():
        if not k.startswith('wandb'):
            print(f"{k}: {v}")
    
    # Run 9-fold cross-validation
    best_model, avg_metrics, all_fold_metrics = run_cross_validation(config)
    
    # Save best model from CV
    torch.save(best_model, os.path.join(output_dir, 'model.pt'))

    
    print("\n==== Cross-Validation Complete ====")
    print(f"View: {args.view}")
    print(f"Model: {args.model_type}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

