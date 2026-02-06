"""
Trainer for GNN + LSTM Vulnerability Detection Model

This module provides a comprehensive training pipeline including:
- Training loop with validation
- Checkpointing and early stopping
- Learning rate scheduling  
- TensorBoard logging
- Metrics tracking
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

from ..models.combined import CombinedModel
from ..data.dataset import VulnerabilityDataset, collate_fn
from .metrics import MetricsCalculator, calculate_batch_metrics
from .config import TrainingConfig, ModelConfig


class Trainer:
    """
    Trainer for vulnerability detection model
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: CombinedModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Combined GNN + LSTM model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Loss function
        if config.class_weights is not None:
            weights = torch.tensor(config.class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:  # sgd
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        
        # Learning rate scheduler
        if config.scheduler == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs
            )
        else:  # step
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        
        # Metrics
        self.metrics_calculator = MetricsCalculator(num_classes=2)
        
        # Tensorboard
        self.writer = SummaryWriter(config.tensorboard_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            (node_features, edge_index, batch_tensor), token_ids, attention_mask, labels = batch
            
            # Move to device
            node_features = node_features.to(self.device)
            edge_index = edge_index.to(self.device)
            batch_tensor = batch_tensor.to(self.device)
            token_ids = token_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(
                node_features, edge_index, batch_tensor,
                token_ids, attention_mask
            )
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(labels.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if self.global_step % self.config.log_every_n_steps == 0:
                self.writer.add_scalar(
                    'train/batch_loss',
                    loss.item(),
                    self.global_step
                )
            
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_probabilities = torch.cat(all_probabilities)
        
        metrics = self.metrics_calculator.calculate_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        return avg_loss, metrics
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate model
        
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                (node_features, edge_index, batch_tensor), token_ids, attention_mask, labels = batch
                
                # Move to device
                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)
                batch_tensor = batch_tensor.to(self.device)
                token_ids = token_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(
                    node_features, edge_index, batch_tensor,
                    token_ids, attention_mask
                )
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Track metrics
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
                all_probabilities.append(probabilities.cpu())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_probabilities = torch.cat(all_probabilities)
        
        metrics = self.metrics_calculator.calculate_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        return avg_loss, metrics
    
    def train(self):
        """
        Main training loop
        """
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.config.scheduler == 'reduce_on_plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            self.writer.add_scalar(
                'lr',
                self.optimizer.param_groups[0]['lr'],
                epoch
            )
            
            # Print results
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_loss = val_loss
                self.save_checkpoint(self.config.best_model_path, is_best=True)
                print(f"  [BEST] New best model! Val F1: {self.best_val_f1:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        print(f"\n Training complete!")
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        self.writer.close()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': self.config.__dict__
        }
        
        if is_best:
            torch.save(checkpoint, filename)
        else:
            path = os.path.join(self.config.checkpoint_dir, filename)
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint['best_val_f1']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
