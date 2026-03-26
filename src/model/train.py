"""
Training script for Sign Language Recognition

Works with directory structure:
    data_dir/
        train/<gloss>/*.mp4 (or *.npy)
        val/<gloss>/*.mp4 (or *.npy)
        test/<gloss>/*.mp4 (or *.npy)

Features:
- Runtime augmentation
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Proper logging
"""

import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Import from project
from src.model.data_loader import SignLanguageDataset, create_data_loaders
from src.utils.logger import *
from src.config.config import *


def str2bool(value):
    """Parse common boolean string representations for CLI arguments."""
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y", "on"}:
        return True
    if val in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

# =====================================================
# Models
# =====================================================

class SimpleLSTM(nn.Module):
    """Simple LSTM for small datasets"""
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=10, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.classifier(out[:, -1, :])


class BiLSTM(nn.Module):
    """Bidirectional LSTM"""
    def __init__(self, input_dim=225, hidden_dim=128, num_layers=2, 
                 num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.classifier(out[:, -1, :])


class SimpleGRU(nn.Module):
    """Simple GRU for very small datasets"""
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=10, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out, hn = self.gru(x)
        return self.classifier(out[:, -1, :])


def build_model(model_type, input_dim, hidden_dim, num_classes, 
                num_layers, dropout, bidirectional=False):
    """Build model based on type"""
    if model_type not in ['lstm', 'bilstm', 'gru']:
        raise ValueError(f"Unknown model type: {model_type}")
    if num_classes <= 0 or num_layers <= 0:
        raise ValueError("num_classes, num_layers must be specified and greater than 0")
    if model_type == 'lstm':
        return SimpleLSTM(input_dim, hidden_dim, num_classes, dropout)
    elif model_type == 'bilstm':
        return BiLSTM(input_dim, hidden_dim, num_layers, num_classes, dropout)
    elif model_type == 'gru':
        return SimpleGRU(input_dim, hidden_dim, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =====================================================
# Training Utilities
# =====================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        return self.early_stop


def save_checkpoint(model, optimizer, epoch, path, **extra):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        **extra
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint


# =====================================================
# Main Training Function
# =====================================================

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Print configuration
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70 + "\n")
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    # Build runtime augmentation config from CLI
    augment_config = {
        'rotation_range': args.rotation_range,
        'scale_range': (args.scale_min, args.scale_max),
        'shift_range': args.shift_range,
        'flip_prob': args.flip_prob,
        'time_mask_prob': args.time_mask_prob,
        'time_mask_max': args.time_mask_max,
    }

    # Create datasets
    train_ds = SignLanguageDataset(
        args.data_dir,
        seq_len=args.seq_len,
        source=args.source,
        split='train',
        normalize=args.normalize,
        augment=args.augment_train,
        augment_config=augment_config,
    )
    
    label_map = train_ds.get_label_map()
    num_classes = len(label_map)
    
    val_ds = SignLanguageDataset(
        args.data_dir,
        seq_len=args.seq_len,
        source=args.source,
        split='val',
        normalize=args.normalize,
        augment=False,  # No augmentation for validation
        label_map=label_map,
    )
    
    logger.info(f"Training samples: {len(train_ds)}")
    logger.info(f"Validation samples: {len(val_ds)}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {label_map}\n")
    
    # Save label map
    label_map_path = os.path.join(args.ckpt_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f)
    logger.info(f"Label map saved -> {label_map_path}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # =========================================================================
    # Data Sanity Check
    # =========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("DATA SANITY CHECK")
    logger.info("=" * 70)
    
    for i in range(min(3, len(train_ds))):
        X, y = train_ds[i]
        logger.info(f"Sample {i}: shape={X.shape}, label={y.item()} ({label_map[y.item()]}), "
              f"range=[{X.min():.3f}, {X.max():.3f}], "
              f"mean={X.mean():.3f}, std={X.std():.3f}")
        
        if torch.all(X == 0):
            logger.warning(f"  ⚠️  WARNING: Sample {i} is all zeros!")
        if torch.any(torch.isnan(X)):
            logger.error(f"  ❌ ERROR: Sample {i} contains NaN!")
    
    # =========================================================================
    # Model
    # =========================================================================
    
    model = build_model(
        args.model_type,
        args.input_dim,
        args.hidden_dim,
        num_classes,
        args.num_layers,
        args.dropout,
        args.bidirectional,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel: {args.model_type.upper()}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # =========================================================================
    # Loss, Optimizer, Scheduler
    # =========================================================================
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )
    
    early_stopping = EarlyStopping(patience=args.patience)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, args.epochs + 1):
        # ----- Training -----
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        t0 = time.time()
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            
            # Forward
            logits = model(X)
            loss = criterion(logits, y)
            
            # Check for NaN
            if torch.isnan(loss):
                logger.info(f"❌ NaN loss at epoch {epoch}, batch {batch_idx}")
                return
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            train_labels.extend(y.cpu().numpy().tolist())
        
        avg_train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # ----- Validation -----
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                
                logits = model(X)
                loss = criterion(logits, y)
                
                val_losses.append(loss.item())
                val_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                val_labels.extend(y.cpu().numpy().tolist())
        
        avg_val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_labels, val_preds)
        
        elapsed = time.time() - t0
        
        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            ckpt_path = os.path.join(args.ckpt_dir, "best.pth")
            save_checkpoint(
                model, optimizer, epoch, ckpt_path,
                val_acc=val_acc,
                label_map=label_map,
                args=vars(args),
            )
            logger.info(f"  ✓ New best! Saved -> {ckpt_path}")
        
        # Early stopping
        if early_stopping(val_acc):
            logger.info(f"\nEarly stopping at epoch {epoch}")
            break
        
        # Debug: Print prediction distribution
        if epoch % 10 == 0:
            logger.info(f"  Pred dist: {np.bincount(val_preds, minlength=num_classes)}")
            logger.info(f"  True dist: {np.bincount(val_labels, minlength=num_classes)}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    logger.info(f"Model saved to: {os.path.join(args.ckpt_dir, 'best.pth')}")
    
    # Save training history
    history_path = os.path.join(args.ckpt_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    logger.info(f"Training history saved to: {history_path}")
    
    # Evaluate on test set if available
    test_dir = os.path.join(args.data_dir, 'test')
    if os.path.exists(test_dir):
        logger.info("\n" + "=" * 70)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 70)
        
        test_ds = SignLanguageDataset(
            args.data_dir,
            seq_len=args.seq_len,
            source=args.source,
            split='test',
            normalize=args.normalize,
            augment=False,
            label_map=label_map,
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        
        # Load best model
        load_checkpoint(
            os.path.join(args.ckpt_dir, 'best.pth'),
            model,
            device=device
        )
        
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                logits = model(X)
                test_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                test_labels.extend(y.cpu().numpy().tolist())
        
        test_acc = accuracy_score(test_labels, test_preds)
        logger.info(f"\nTest Accuracy: {test_acc:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(test_labels, test_preds, target_names=label_map))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    
    # Data
    parser.add_argument('--data_dir', default='data/wlasl/wlasl100',
                        help='Root directory with train/val/test subdirs')
    parser.add_argument('--source', choices=['npy', 'video'], default='npy',
                        help='Load from .npy files or extract from videos')
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN,
                        help='Fixed sequence length')
    parser.add_argument('--input_dim', type=int, default=INPUT_DIM,
                        help='Input feature dimension (225 for pose+hands)')
    
    # Model
    parser.add_argument('--model_type', choices=['lstm', 'bilstm', 'gru'], 
                        default=MODEL_TYPE, help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional RNN')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.2,
                        help='Label smoothing')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Data loading workers')

    # Data processing
    parser.add_argument('--normalize', type=str2bool, default=True,
                        help='Enable keypoint normalization (true/false)')
    parser.add_argument('--augment_train', type=str2bool, default=True,
                        help='Enable runtime augmentation on training split (true/false)')
    parser.add_argument('--rotation_range', type=float, default=15.0,
                        help='Max rotation in degrees for augmentation')
    parser.add_argument('--scale_min', type=float, default=0.85,
                        help='Min scale factor for augmentation')
    parser.add_argument('--scale_max', type=float, default=1.15,
                        help='Max scale factor for augmentation')
    parser.add_argument('--shift_range', type=float, default=0.08,
                        help='Max XY translation range for augmentation')
    parser.add_argument('--flip_prob', type=float, default=0.5,
                        help='Probability of horizontal flip augmentation')
    parser.add_argument('--time_mask_prob', type=float, default=0.2,
                        help='Probability of random temporal masking augmentation')
    parser.add_argument('--time_mask_max', type=int, default=3,
                        help='Max consecutive frames to mask in time masking')
    
    # Output
    parser.add_argument('--ckpt_dir', default='models/checkpoints',
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    logger = setup_logger("train")
    log_arguments(logger, args)
    train(args)