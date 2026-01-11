import random
import requests
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

import unittest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from torch.utils.data import TensorDataset


# =============================================================================
# DATASET & MODEL
# =============================================================================

class CIFAR100Filtered(Dataset):
    """
    CIFAR-100 dataset wrapper with preprocessing and train/val split support.
    
    Args:
        root (str): Directory to store/load CIFAR-100 data
        split (str): Either "train" or "val" to specify which split to use
        transform (callable, optional): Transform to apply to images. If None, uses default.
    
    Attributes:
        dataset: The underlying torchvision CIFAR100 dataset
    """
    
    def __init__(self, root="./data", split="train", transform=None): 
                
        assert split in ("train", "val"), "split must be in 'train' or 'val.'"

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std  = [0.229, 0.224, 0.225],
                )
            ])
 
        self.dataset = datasets.CIFAR100(root=root, train=(split=="train"), download=True,
                                        transform=transform)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: (image, label) where image is a transformed tensor and label is an integer
        """ 
        return self.dataset[idx]



class ImageEncoder(nn.Module):
    """
    MobileNetV3-based image encoder with trainable projection head.
    
    This model consists of two parts:
    1. Frozen pretrained MobileNetV3 backbone (feature extractor)
    2. Trainable projection head (maps features to embedding space)
    
    Args:
        proj_dim (int): Dimension of the output projection embeddings
        device (str): Device to place the model on ("cuda" or "cpu")
    
    Attributes:
        backbone: Frozen MobileNetV3 feature extractor (output: 576-dim)
        projection: Trainable MLP that projects to proj_dim
    """
    
    def __init__(self, proj_dim=64, device="cuda"):
        super().__init__()
        self.device = device 
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone = nn.Sequential(*list(base.children())[:-1]).to(device).eval()
        
        self.backbone = backbone
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(576, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        ).to(device)
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            tuple: (backbone_features, projected_embeddings)
                - backbone_features: Raw features from MobileNet (batch_size, 576)
                - projected_embeddings: Projected embeddings (batch_size, proj_dim)
        """ 
        with torch.no_grad():
            features = self.backbone(x).flatten(1)
        output = self.projection(features)

        return features, output
    
# =============================================================================
# DATA & TRAINING UTILITIES
# =============================================================================

def filter_dataset_indices(dataset, valid_labels):
    """Return indices of samples with labels in valid_labels set."""
    return [i for i, label in enumerate(dataset.dataset.targets) if label in valid_labels]

def create_data_splits(indices, val_ratio=0.2, seed=42):
    """Split indices into train/val sets."""
    np.random.seed(seed)
    indices = np.array(indices)
    np.random.shuffle(indices)
    split_idx = int((1 - val_ratio) * len(indices))
    return indices[:split_idx].tolist(), indices[split_idx:].tolist()

def create_dataloaders(train_idx, val_idx, test_idx, batch_sizes):
    """Create train, val, and test dataloaders."""
    datasets = {
        'train': Subset(CIFAR100Filtered(split="train"), train_idx),
        'val': Subset(CIFAR100Filtered(split="train"), val_idx), 
        'test': Subset(CIFAR100Filtered(split="val"), test_idx)
    }
    return {k: DataLoader(v, batch_size=batch_sizes['train' if k == 'train' else 'eval'], 
                         shuffle=(k == 'train'), num_workers=2) for k, v in datasets.items()}



def compute_contrastive_loss(visual_proj, text_emb, temperature):
    """
    Compute symmetric InfoNCE (contrastive) loss for vision-language alignment.
    
    This loss encourages matching pairs (image, correct_text) to have high similarity
    while pushing apart non-matching pairs (image, wrong_text).
    
    Args:
        visual_proj (torch.Tensor): Projected visual embeddings, shape (batch_size, proj_dim)
        text_emb (torch.Tensor): Text embeddings for the batch, shape (batch_size, proj_dim)
        temperature (float): Temperature parameter to scale logits (typically 0.07)
    
    Returns:
        torch.Tensor: Scalar loss value (symmetric InfoNCE loss)
    
    Mathematical formulation:
        1. Normalize both embeddings to unit vectors
        2. Compute similarity matrix: S = (visual @ text.T) / temperature
        3. Apply cross-entropy loss treating diagonal as correct matches
        4. Average image-to-text and text-to-image losses for symmetry
    """ 
    
    V = F.normalize(visual_proj, p=2, dim=1)
    S = F.normalize(text_emb, p=2, dim=1) 

    logits = torch.matmul(V, S.T) / temperature
    labels = torch.arange(V.shape[0], device=visual_proj.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


def run_epoch(model, dataloader, text_emb, class_words, label_to_word, optimizer, temperature, device, mode='train'):
    """
    Run one epoch of training or evaluation for contrastive vision-language learning.
    
    Handles a single pass over the dataset, computes loss and mean similarity,
    and updates the model if in training mode.
    
    Args:
        model (nn.Module): Image encoder model.
        dataloader (DataLoader): DataLoader providing (image, label) batches.
        text_emb (torch.Tensor): Embeddings for all class words, shape (num_classes, proj_dim).
        class_words (list): List of word strings for each class label index.
        label_to_word (dict): Maps integer label to the corresponding word string.
        optimizer (torch.optim.Optimizer or None): Optimizer for model parameters (set to None in eval mode).
        temperature (float): Contrastive loss temperature parameter.
        device (str or torch.device): Device for computation.
        mode (str): 'train' or 'eval' (evaluation).
    
    Returns:
        tuple: (mean_loss, mean_similarity)
            mean_loss: Average loss over the epoch.
            mean_similarity: Average cosine similarity between visual and aligned text embeddings.
    """
 
    is_train = (mode == 'train')
    model.train() if is_train else model.eval() 
    assert len(class_words) == text_emb.shape[0]
    
    text_emb = text_emb.to(device)
    total_loss = 0
    total_sim = 0
    count = 0
 
    with torch.no_grad() if mode == 'eval' else torch.enable_grad():
        for images, labels in tqdm(dataloader, desc=f"[{mode.upper}]" if mode == 'train' else None):
            images, labels= images.to(device), labels.to(device)
            _, visual_proj = model(images)
            
            batch_text_idx = torch.tensor([label_to_word[label.item()] for label in labels], dtype=torch.long, device=device)
            batch_text_emb = text_emb[batch_text_idx]

            loss = compute_contrastive_loss(visual_proj, text_emb[labels], temperature)
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_sim += torch.sum(F.normalize(visual_proj, p=2, dim=1) * F.normalize(batch_text_emb, p=2, dim=1)).item()
            count += images.size(0)
    return total_loss / count, total_sim / count


def run_epoch(model, dataloader, text_emb, class_words, label_to_word,
              optimizer, temperature, device, mode='train'):
    """
    Run one epoch of training or evaluation for contrastive vision-language learning.
    Returns: (mean_loss, mean_similarity)
    """
    is_train = (mode == 'train')
    model.train() if is_train else model.eval()

    # text_emb is expected to be aligned with class_words (same order)
    text_emb = text_emb.to(device)

    # Map CIFAR label -> index into text_emb (via class_words)
    # label_to_word: {label_int: "word"}
    label_to_text_index = {}
    for lbl, w in label_to_word.items():
        if w in class_words:
            label_to_text_index[int(lbl)] = class_words.index(w)

    total_loss = 0.0
    total_sim = 0.0
    count = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    loop = tqdm(dataloader, desc=f"[{mode.upper()}]" if mode == 'train' else None)
    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Build batch text embeddings in the same order as images
        # (each image pairs with its class word embedding)
        text_idx = torch.tensor(
            [label_to_text_index[int(l)] for l in labels.tolist()],
            dtype=torch.long,
            device=device
        )
        batch_text_emb = text_emb[text_idx]  # (B, proj_dim)

        with ctx:
            _, visual_proj = model(images)  # (B, proj_dim)
            loss = compute_contrastive_loss(visual_proj, batch_text_emb, temperature)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Mean cosine similarity for matched pairs in the batch
        with torch.no_grad():
            v = F.normalize(visual_proj, p=2, dim=1)
            t = F.normalize(batch_text_emb, p=2, dim=1)
            batch_sim_sum = (v * t).sum(dim=1).sum().item()  # sum over batch

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_sim += batch_sim_sum
        count += bs

        loop.set_postfix(loss=loss.item())

    return total_loss / max(count, 1), total_sim / max(count, 1)
 

def train_with_early_stopping(model, dataloaders, text_emb, class_words, label_to_word, config, device):
    """Train model with early stopping and return history."""
    optimizer = torch.optim.AdamW(model.projection.parameters(),
                                lr=config['lr'], 
                                weight_decay=config['weight_decay']
                            )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_val_sim = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = defaultdict(list)
    
    print(f"\n{'='*70}\nTraining (max {config['epochs']} epochs, patience={config['patience']})\n{'='*70}")
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            train_loss, _ = run_epoch(model, 
                                    dataloaders['train'],
                                    text_emb,
                                    class_words,
                                    label_to_word,
                                    optimizer,
                                    config['temperature'],
                                    device,
                                    'train'
                                )
            val_loss, val_sim = run_epoch(model, 
                                        dataloaders['val'],
                                        text_emb,
                                        class_words,
                                        label_to_word,
                                        None,
                                        config['temperature'],
                                        device,
                                        'eval'
                                    )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Update history
            for metric, value in zip(['train_loss', 'val_loss', 'val_similarity', 'learning_rate'], 
                                    [train_loss, val_loss, val_sim, current_lr]):
                history[metric].append(value)
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Sim: {val_sim:.4f} | LR: {current_lr:.6f}")
            
            if val_sim > best_val_sim:
                best_val_sim, best_val_loss, best_epoch, patience_counter = val_sim, val_loss, epoch, 0
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss,
                    'val_similarity': val_sim, 'class_words': class_words, 'text_embeddings': text_emb.cpu(),
                    'history': dict(history), 'projection_head': model.projection.state_dict(),
                }, config['save_path'])
                print(f"  ✓ New best model saved! (Val Sim: {val_sim:.4f})")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{config['patience']})")
            
            if patience_counter >= config['patience']:
                print(f"\n{'='*70}\nEarly stopping at epoch {epoch}\nBest: {best_epoch} (Val Sim: {best_val_sim:.4f})\n{'='*70}")
                break
    except Exception as e:
        print(f"\n⚠️  Training crashed with error: {str(e)}")
        print(f"Attempting to continue with best saved model...")
    
    return dict(history), best_epoch, best_val_sim, best_val_loss



# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def collect_embeddings(model, dataloader, device):
    """Collect all embeddings and labels from dataset."""
    model.eval()
    all_visual, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting embeddings"):
            images = images.to(device)
            _, visual_proj = model(images)
            all_visual.append(F.normalize(visual_proj, p=2, dim=1).cpu())
            all_labels.extend(labels.tolist())
    return torch.cat(all_visual, dim=0).numpy(), all_labels


def compute_alignment_metrics(visual_emb, labels, text_emb, class_words, label_to_word):
    """Compute comprehensive alignment metrics in one pass."""
    # Per-class statistics
    class_sims = defaultdict(list)
    for i, label in enumerate(labels):
        if (word := label_to_word[label]) in class_words:
            sim = np.dot(visual_emb[i], text_emb[class_words.index(word)])
            class_sims[word].append(sim)
    
    stats = sorted([{
        'word': word, 'mean': np.mean(sims), 'std': np.std(sims), 
        'min': np.min(sims), 'max': np.max(sims), 'count': len(sims)
    } for word, sims in class_sims.items()], key=lambda x: x['mean'], reverse=True)
    
    # Retrieval metrics
    sim_matrix = cosine_similarity(visual_emb, text_emb)
    i2t_recalls = {k: 0 for k in [1, 5, 10]}
    t2i_recalls = {k: 0 for k in [1, 5, 10]}
    
    # Image-to-text retrieval
    for i, label in enumerate(labels):
        if (word := label_to_word[label]) in class_words:
            correct_idx = class_words.index(word)
            ranking = np.argsort(-sim_matrix[i])
            for k in i2t_recalls:
                if correct_idx in ranking[:k]: i2t_recalls[k] += 1
    
    # Text-to-image retrieval  
    for class_idx, word in enumerate(class_words):
        class_img_idx = [i for i, l in enumerate(labels) if label_to_word[l] == word]
        if class_img_idx:
            ranking = np.argsort(-sim_matrix[:, class_idx])
            for k in t2i_recalls:
                if any(idx in ranking[:k] for idx in class_img_idx): t2i_recalls[k] += 1
    
    return stats, i2t_recalls, t2i_recalls, sim_matrix


def print_analysis_results(stats, i2t_recalls, t2i_recalls, n_samples, n_classes):
    """Print comprehensive analysis results."""
    print("\n📊 Per-Class Similarity Analysis:")
    print("-" * 70)
    for title, data in [("Top 10 Best Aligned Classes:", stats[:10]), 
                        ("Bottom 10 Worst Aligned Classes:", stats[-10:])]:
        print(f"\n{title}")
        for i, s in enumerate(data, 1):
            print(f"{i:2d}. {s['word']:15s} | Mean: {s['mean']:.4f} ± {s['std']:.4f}")
    
    print("\n📊 Retrieval Performance:")
    print("-" * 70)
    for name, recalls, total in [("Image-to-Text", i2t_recalls, n_samples), 
                                 ("Text-to-Image", t2i_recalls, n_classes)]:
        print(f"\n{name} Retrieval (Recall@K):")
        for k, count in recalls.items():
            print(f"  Recall@{k:2d}: {count/total*100:.2f}% ({count}/{total})")

def print_example_retrievals(sim_matrix, labels, class_words, label_to_word, n_examples=5):
    """Print text-based retrieval examples."""
    print("\n📸 Example Image-to-Text Retrievals:")
    print("-" * 70)
    
    display_idx = np.random.choice(len(labels), size=n_examples, replace=False)
    
    for idx in display_idx:
        label = labels[idx]
        true_word = label_to_word[label]
        
        sims = sim_matrix[idx]
        top_5_idx = np.argsort(-sims)[:5]
        top_5_words = [class_words[i] for i in top_5_idx]
        top_5_sims = [sims[i] for i in top_5_idx]
        
        correct_sim = sims[class_words.index(true_word)]
        correct_rank = np.where(np.argsort(-sims) == class_words.index(true_word))[0][0] + 1
        
        print(f"\nTest Image #{idx}:")
        print(f"  True class: '{true_word}' (similarity: {correct_sim:.4f}, rank: {correct_rank})")
        print(f"  Top 5 predictions:")
        for rank, (word, sim) in enumerate(zip(top_5_words, top_5_sims), 1):
            marker = "✓" if word == true_word else " "
            print(f"    {rank}. {marker} {word:15s} (similarity: {sim:.4f})")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_visualizations(sim_matrix, labels, class_words, label_to_word, test_indices, images=None, names=None, predictions=None):
    """Create all visualizations in one coordinated function."""
    # OOD analysis if provided
    if images and names and predictions:
        print(f"\n📸 Creating OOD visualization for {len(images)} images...")
        n_imgs = len(images)
        n_cols = min(4, n_imgs)
        n_rows = (n_imgs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 7*n_rows))
        axes = [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        
        for i, (img, name, pred) in enumerate(zip(images, names, predictions)):
            axes[i].imshow(img); axes[i].axis('off')
            pred_text = f"{name.upper()}\n\nTop matches:\n"
            for rank, (word, sim) in enumerate(zip(pred['words'][:5], pred['sims'][:5]), 1):
                pred_text += f"{rank}. {word} ({sim:.3f})\n"
            axes[i].set_title(pred_text, fontsize=11, ha='center', color='darkblue', fontweight='bold', pad=12)
        
        for j in range(i+1, len(axes)): 
            axes[j].axis('off'); axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('ood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    else:
        print("\n📊 Creating confusion matrix...")
        n_classes = len(class_words)
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for i, label in enumerate(labels):
            if (word := label_to_word[label]) in class_words:
                true_idx = class_words.index(word)
                pred_idx = np.argmax(sim_matrix[i])
                conf_matrix[true_idx, pred_idx] += 1
        
        conf_matrix = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(conf_matrix, xticklabels=class_words, yticklabels=class_words,
                    cmap='Blues', ax=ax, cbar_kws={'label': 'Probability'}, square=True)
        ax.set_xlabel('Predicted Class'); ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix (All Classes)', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Retrieval examples
        print("\n📸 Creating retrieval examples...")
        test_raw = CIFAR100Filtered(split="val", transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for plot_idx, ax in enumerate(axes.flatten()):
            if plot_idx >= 12: break
            ex_idx = random.randint(0, len(labels)-1)
            original_idx = test_indices[ex_idx]
            img, label = test_raw[original_idx]
            
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis('off')
            
            true_word = label_to_word[label]
            sims = sim_matrix[ex_idx]
            top_5_idx = np.argsort(-sims)[:5]
            top_5_words = [class_words[i] for i in top_5_idx]
            top_5_sims = [sims[i] for i in top_5_idx]

            # Build prediction text
            pred_text = f"GT: {true_word}\n"
            for rank, (word, sim) in enumerate(zip(top_5_words, top_5_sims), 1):
                marker = "✓" if word == true_word else "✗"
                pred_text += f"{rank}. {marker} {word}: {sim:.2f}\n"

            # --- COLOR LOGIC CHANGE ---
            if top_5_words[0] == true_word:
                title_color = "green"          # correct top-1
            elif true_word in top_5_words:
                title_color = "#CC8A00"        # amber
            else:
                title_color = "red"            # incorrect
            # --------------------------

            ax.set_title(
                pred_text, fontsize=9, ha='left',
                fontfamily='monospace',
                color=title_color, fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig('retrieval_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        

# =============================================================================
# OOD PROCESSING
# =============================================================================

def process_ood_images(model, image_urls, text_emb, class_words, device):
    """Download and process OOD images in one function."""
    print(f"\nDownloading {len(image_urls)} OOD test images...")
    images, names, headers = [], [], {'User-Agent': 'Mozilla/5.0', 'Accept': 'image/*'}
    
    for desc, url in image_urls.items():
        try:
            response = requests.get(url, timeout=30, headers=headers)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                images.append(img.resize((224, 224), Image.BILINEAR))
                names.append(desc)
                print(f"  ✓ Downloaded: {desc}")
        except Exception as e:
            print(f"  ✗ Error downloading {desc}: {str(e)[:50]}")
    
    if not images: return [], [], []
    
    print(f"\n🔬 Processing {len(images)} OOD images...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    model.eval()
    with torch.no_grad():
        ood_emb = []
        for img in images:
            img_tensor = normalize(to_tensor(img).unsqueeze(0).to(device))
            _, visual_proj = model(img_tensor)
            ood_emb.append(F.normalize(visual_proj, p=2, dim=1).cpu().numpy()[0])
    
    ood_emb = np.array(ood_emb)
    predictions = []
    for emb in ood_emb:
        sims = cosine_similarity(emb.reshape(1, -1), text_emb)[0]
        top_5_idx = np.argsort(-sims)[:5]
        predictions.append({
            'words': [class_words[j] for j in top_5_idx],
            'sims': [sims[j] for j in top_5_idx]
        })
    
    return images, names, predictions



# =============================================================================
# PREDICT A SINGLE IMAGE
# =============================================================================



# =============================================================================
# FINAL REPORTING
# =============================================================================

def print_final_report(config, test_loss, test_sim, i2t_recalls, t2i_recalls, class_stats,
                      n_train, n_val, n_test, n_classes, batch_size, has_ood, history=None, 
                      best_epoch=None, best_val_sim=None, best_val_loss=None, n_vocab_total=None):
    """Print comprehensive final summary report."""
    n_samples = n_test
    
    print(f"""
📋 Training Configuration:
   ├─ Model: MobileNetV3-Small with projection head
   ├─ Embedding dimension: {config['proj_dim']}
   ├─ Training samples: {n_train:,}, Validation samples: {n_val:,}, Test samples: {n_test:,}
   ├─ Number of training classes: {n_classes}
   {f'├─ Total vocabulary size: {n_vocab_total} words' if n_vocab_total else ''}
   {f'├─ Total epochs trained: {len(history["train_loss"]) if history else 0}, Best epoch: {best_epoch if best_epoch else "N/A"}' if history else ''}
   └─ Early stopping patience: {config['patience']}

🎯 Performance Metrics:
   {f"├─ Best Val Similarity: {best_val_sim:.4f}" if best_val_sim else ""}
   ├─ Test Similarity: {test_sim:.4f}, Test Loss: {test_loss:.4f}
   ├─ Random baseline loss: ~{np.log(batch_size):.2f}
   │
   ├─ Image→Text Recall@1: {i2t_recalls[1]/n_samples*100:.2f}%
   ├─ Image→Text Recall@5: {i2t_recalls[5]/n_samples*100:.2f}%
   ├─ Image→Text Recall@10: {i2t_recalls[10]/n_samples*100:.2f}%
   │
   ├─ Text→Image Recall@1: {t2i_recalls[1]/n_classes*100:.2f}%
   ├─ Text→Image Recall@5: {t2i_recalls[5]/n_classes*100:.2f}%
   └─ Text→Image Recall@10: {t2i_recalls[10]/n_classes*100:.2f}%

📊 Embedding Space Alignment:
   ├─ Mean per-class similarity: {np.mean([s['mean'] for s in class_stats]):.4f} ± {np.std([s['mean'] for s in class_stats]):.4f}
   ├─ Best aligned class: '{class_stats[0]['word']}' ({class_stats[0]['mean']:.4f})
   └─ Worst aligned class: '{class_stats[-1]['word']}' ({class_stats[-1]['mean']:.4f})

💡 Key Insights:
   • The model {'successfully learns' if test_sim > 0.5 else 'attempts to learn'} visual-text alignment
   • {'High' if test_sim > 0.7 else 'Moderate' if test_sim > 0.5 else 'Low'} overall alignment (similarity: {test_sim:.4f})
   • Loss: {test_loss:.2f} vs random baseline ~{np.log(batch_size):.2f}
   • Retrieval performance: {'Good' if i2t_recalls[1]/n_samples > 0.5 else 'Moderate'}
   • Class performance varies (range: {class_stats[-1]['mean']:.4f} to {class_stats[0]['mean']:.4f})
   {f'• OOD predictions use full vocabulary of {n_vocab_total} words' if n_vocab_total else ''}

✅ Model saved to: '{config['save_path']}'
✅ Confusion matrix & retrieval examples saved
{'✅ OOD analysis saved' if has_ood else ''}
""")
