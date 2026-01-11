"""
Lab 6: Skip-Gram with Negative Sampling (SGNS) for Network Embeddings

Implements Skip-Gram with Negative Sampling to learn embeddings from text networks.
Includes training, evaluation, and visualization tools.

KEY FEATURES:
1. Filters punctuation tokens to prevent hub poisoning
2. Proper negative sampling (5-20 negatives per positive)
3. Weighted sampling by co-occurrence frequency
4. Anti-overfitting: dropout, weight decay, label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import networkx as nx
import requests
import zipfile
import json
import os
from typing import List, Dict, Set, Tuple

import io
import contextlib
import unittest
from collections import Counter


# ============================================================================
# Utilities
# ============================================================================

def download_file(url, out_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {out_path}")


def prepare_visual_genome_text(zip_url, zip_path="region_descriptions.json.zip", 
                                json_path="region_descriptions.json",
                                output_path="vg_text.txt"):
    """Download, unzip, and process Visual Genome region descriptions."""
    
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping processing.")
        return output_path

    if not os.path.exists(zip_path):
        download_file(zip_url, zip_path)
    
    if not os.path.exists(json_path):
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    print(f"Processing {json_path} into {output_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    phrases = [region['phrase'] for img in data for region in img['regions']]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(" . ".join(phrases))
    
    print(f"Processed {len(phrases):,} phrases into {output_path}")
    return output_path
 
 
def filter_punctuation_from_network(network_data, punctuation_tokens={'.', ',', '<RARE>', "'"}):
    """
    Remove punctuation tokens from network to prevent hub poisoning.
    
    Punctuation creates massive hubs that bridge unrelated sentences,
    poisoning the graph structure and making embeddings meaningless.
    """
    original_graph = network_data['graph']
    original_nodes = network_data['nodes']
    original_distance_matrix = network_data['distance_matrix']
    
    # Filter nodes
    filtered_nodes = [n for n in original_nodes if n not in punctuation_tokens]
    old_indices = [i for i, n in enumerate(original_nodes) if n not in punctuation_tokens]
    
    # Filter matrices
    filtered_distance_matrix = original_distance_matrix[np.ix_(old_indices, old_indices)]
    
    # Create filtered graph
    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(filtered_nodes)
    for u, v in original_graph.edges():
        if u in filtered_nodes and v in filtered_nodes:
            filtered_graph.add_edge(u, v)
    
    print(f"\n🔧 PUNCTUATION FILTER:")
    print(f"  Removed: {punctuation_tokens}")
    print(f"  Nodes: {len(original_nodes):,} → {len(filtered_nodes):,}")
    print(f"  Edges: {original_graph.number_of_edges():,} → {filtered_graph.number_of_edges():,}")
    
    return {
        **network_data,
        'graph': filtered_graph,
        'nodes': filtered_nodes,
        'distance_matrix': filtered_distance_matrix
    }


# ============================================================================
# Dataset
# ============================================================================


"""
SkipGramDataset
=======================================

LEARNING OBJECTIVES:
1. Understand how to build a PyTorch Dataset from graph/network data
2. Learn to implement weighted sampling for training pairs
3. Master negative sampling techniques for contrastive learning
4. Handle multi-worker DataLoader scenarios with proper RNG seeding

WHAT YOU'LL IMPLEMENT:
- [ ] _build_contexts(): Extract graph neighborhoods 
- [ ] _generate_weighted_pairs(): Create training pairs with importance weights
- [ ] getitem(): Sample negatives on-the-fly with proper exclusions

TESTING YOUR CODE:
Run the unit tests at the bottom to verify your implementation:
    python skipgram_dataset.py
"""

class SkipGramDataset(torch.utils.data.Dataset):
    """
    Skip-Gram dataset for learning node embeddings from a graph structure.
    
    This dataset:
    - Builds (center, context) training pairs from graph neighbors
    - Computes importance weights for weighted sampling
    - Samples negative examples on-the-fly during training
    - Handles multi-worker data loading with independent random streams
    
    Example Usage:
        >>> graph = nx.karate_club_graph()
        >>> nodes = list(graph.nodes())
        >>> dist_matrix = compute_distance_matrix(graph, nodes)  # your function
        >>> dataset = SkipGramDataset(graph, nodes, dist_matrix)
        >>> center, context, negatives = dataset[0]
        >>> print(f"Center: {center}, Context: {context}, Negatives: {negatives[:3]}...")
    """

    def __init__(
        self,
        graph: nx.Graph,
        nodes: List[str],
        distance_matrix: np.ndarray,
        num_negative: int = 15,
        context_size: int = 1,
        verbose = False,
    ):
        """
        Initialize the Skip-Gram dataset.
        
        Args:
            graph: NetworkX graph where nodes are tokens/words
            nodes: Ordered list of node labels (vocabulary)
            distance_matrix: Precomputed distances between nodes, shape (V, V)
            num_negative: Number of negative samples per positive pair (typically 5-20)
            context_size: Context radius in graph hops (1 = immediate neighbors)
        """
        super().__init__()
        
        # Store basic references
        self.graph = graph
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.vocab_size = len(nodes)
        self.num_negative = num_negative
        self.distance_matrix = distance_matrix
        self.verbose = verbose
        
        if distance_matrix.shape != ( len(nodes),len(nodes) ):
            raise ValueError("distance_matrix must be shape (V, V) aligned to nodes")
        else:
            print(f"✅ distance_matrix is shape (V, V) aligned to nodes")


        # Step 1: Build context sets for each node
        self.contexts = self._build_contexts(context_size)
        
        # Step 2: Convert contexts into training pairs and compute weights
        self.pairs, self.weights = self._generate_weighted_pairs()

        self._local_rng = None
        self._print_stats()
 
 
    def _build_contexts(self, context_size: int) -> Dict[str, Set[str]]:
        """
        Build a context set for each node (neighbors within context_size hops).
        
        Algorithm:
        1. For each node in self.nodes:
           a. Use nx.single_source_shortest_path_length() to find all reachable nodes
              within context_size hops
           b. Keep only nodes with distance > 0 (exclude self)
           c. Keep only nodes that exist in self.node_to_idx (vocabulary filter)
        2. Return dict: {node_string: set(neighbor_strings)}
        
        Example:
            If node "cat" has neighbors ["dog", "animal"] within 1 hop:
            contexts["cat"] = {"dog", "animal"}
        
        Args:
            context_size: Maximum number of hops to consider as context
            
        Returns:
            Dictionary mapping each node to its set of context nodes
        """
        contexts = {}
        
        for node in self.nodes:
            if node not in self.graph:
                contexts[node] = set()
                continue
 
            lengths = dict(
                nx.single_source_shortest_path_length(
                    self.graph, source=node, cutoff=context_size
                )
            )
                    
            # Filter to valid vocabulary nodes with distance > 0
            contexts[node] = {
                n for n, d in lengths.items()
                if d > 0 and n in self.node_to_idx
            }
        if self.verbose:
            print(f"✅ _build_contexts() created {len(contexts)} contexts")
            print(f"Now we will generate weighted pairs...\n")
        

        return contexts


    def _generate_weighted_pairs(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Generate (center_idx, context_idx) pairs and compute importance weights.
        
        Algorithm:
        1. Iterate through self.contexts to create pairs
        2. For each (center, context) pair, look up the distance from distance_matrix
        3. Convert distances to weights (closer pairs = higher weight)
        4. Apply transformations to prevent overfitting:
           - Sublinear scaling (sqrt) to reduce extreme weights
           - Clipping to prevent dominance by a few pairs
        5. Normalize weights for interpretability
        

        WEIGHT FORMULA:
        Starting from raw distances (where larger = farther):
            raw_weight = (max_distance + 1) - distance      # invert so closer = larger
            weight = sqrt(raw_weight)                       # sublinear scaling
            weight = clip(weight, max=95th_percentile * 3)  # prevent outliers
            weight = normalize(weight)                      # scale to reasonable range
        
            
        WHY THESE TRANSFORMS:
        - Inversion: Skip-gram should focus on close relationships
        - Sqrt: Prevents a few very-high-frequency pairs from dominating
        - Clipping: Extreme weights can cause training instability
        - Normalization: Makes weights interpretable when printed
        
        Returns:
            pairs: List of (center_idx, context_idx) tuples
            weights: numpy array of weights, same length as pairs
        """
        pairs = []
        raw_distances = []
         
        for center_word, context_words in self.contexts.items():
            center_idx = self.node_to_idx[center_word]
            
            for context_word in context_words:
                context_idx = self.node_to_idx[context_word]
                pairs.append( (center_idx, context_idx) )
                raw_distances.append(self.distance_matrix[center_idx, context_idx])
            
        if len(pairs) == 0:
            return [], np.array([], dtype=np.float32)
        
        raw_distances = np.array(raw_distances, dtype=np.float32)
        
        # Handle any weird infinities just in case
        finite = np.isfinite(raw_distances)
        if not np.any(finite):
            print('Infinities found!')
            raise ValueError('Infinities found!')
        if self.verbose:
            print(f'Shape of raw distances: {raw_distances.shape}') 

        # 2) Convert distances to weights
        max_distance = float(raw_distances.max())
        raw_weight = (max_distance + 1.0) - raw_distances
        raw_weight = np.maximum(raw_weight, 1e-6)  # avoid zeros/negatives
        if self.verbose:
            print(f"\n1. Weights are now created as inverted distances (closer => larger) and are in range [{min(raw_weight):.2f}, {max(raw_weight):.2f}]")
            print(f'Max distance: {max_distance}')
            print(f'Raw weights have shape: {raw_weight.shape}')
        
        # 3) Sublinear scaling
        weights = np.sqrt(raw_weight)
        if self.verbose:
            print(f"\n2. Weights are now sqrt-scaled and are in range [{min(weights):.2f}, {max(weights):.2f}]")
        

        # 4) Clip extreme values
        p95 = float(np.percentile(weights, 95) * 3) 

        clip_threshold = max(p95, 1e-6)
        weights = np.clip(weights, None, clip_threshold)
        if self.verbose:
            print(f"\n3. Weights are now clipped to prevent outliers and are in range [{min(weights):.2f}, {max(weights):.2f}]")
            print(f'95th percentile: {p95}')


        # 5) Normalize (mean = 1 for interpretability) 
        weights = weights / (weights.mean() + 1e-12)
        weights = np.maximum(weights, 1e-6)


        if self.verbose:
            print(f"\n4. Weights are now normalized and are in range [{min(weights):.2f}, {max(weights):.2f}]")
            print(f'Final weights have shape: {weights.shape}')
            print(f'Average weight: {weights.mean():.2f}')
            print(f"✅ _generate_weighted_pairs() created {len(pairs)} pairs and {len(weights)} weights")
 
        idx_to_node = {i:n for n,i in self.node_to_idx.items()}
        topk = 25
        top = np.argsort(weights)[-topk:][::-1]

        print("Top weighted pairs:")
        for i in top:
            c, ctx = pairs[i]
            print(f"{weights[i]:8.3f}  {idx_to_node[c]:>10}  ->  {idx_to_node[ctx]}")

        # are weights too concatretaed? 
        w = weights.astype(np.float64)
        ess = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)
        print("N:", len(w), "ESS:", ess, "ESS/N:", ess/len(w))

                
        return pairs, weights.astype(np.float32)

        

    def __getitem__(self, idx: int) -> Tuple[np.int64, np.int64, np.ndarray]:
        """
        Get a single training example: (center_idx, context_idx, negatives).
        
        Algorithm:
        1. Initialize per-worker RNG if needed (for DataLoader multi-processing)
        2. Retrieve the positive pair at index idx
        3. Build exclusion set: center + all its true contexts
        4. Sample num_negative nodes from vocabulary, excluding the exclusion set
        5. Return (center_idx, context_idx, negatives_array)
        
        WHY EXCLUDE TRUE CONTEXTS:
        If we use a true context node as a "negative" example, we're training
        the model with contradictory signals (it's both positive and negative).
        This confuses learning.
        
        WHY PER-WORKER RNG:
        DataLoader uses multiple worker processes. Each needs an independent
        random stream or they'll all generate identical "random" samples.

        Args:
            idx: Index into self.pairs
            
        Returns:
            center_idx: numpy int64 scalar
            context_idx: numpy int64 scalar  
            negatives: numpy int64 array of shape (num_negative,)
        """
        
        # Step 1 - Initialize per-worker RNG (lazy initialization)
        if self._local_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id

            base_seed = torch.initial_seed()
            # numpy wants a 32-bit-ish seed; mod keeps it in a safe range
            seed = (base_seed + worker_id) % (2**32)
            self._local_rng = np.random.default_rng(seed)
        if self.verbose:
            print(f'\nNow we will get the positive pair at index {idx} with the negatives:')
            print(f'Worker info: {worker_info}')
            print(f'Worker ID: {worker_id}')
            print(f'Base seed: {base_seed}')
            print(f'Seed: {seed}')  
            print(f'Local RNG: {self._local_rng}')


        # Step 2 - Get the positive pair
        center_idx, context_idx = self.pairs[idx]
        if self.verbose:
            print(f'Positive pair at index {idx}: {center_idx} -> {context_idx}')
            print(f'The word pair is {self.nodes[center_idx]} -> {self.nodes[context_idx]}' )

        # Step 3 - Build exclusion set (center + all true contexts of center)
        center_node = self.nodes[center_idx]
        true_contexts = self.contexts.get(center_node, set())
        if self.verbose:
            print(f'Number of true contexts: {len(true_contexts)}')
        
        excluded = {
            self.node_to_idx[w]
            for w in true_contexts
            if w in self.node_to_idx
        }
        excluded.add(center_idx)
        if self.verbose:
            print(f'Excluded num of nodes: {len(excluded)}')

        # Step 4 - Build pool of available negative candidates
        mask = np.ones(self.vocab_size, dtype=bool)
        if excluded:
            mask[np.fromiter(excluded, dtype=np.int64)] = False
        available = np.nonzero(mask)[0].astype(np.int64)

        # Edge case: if nothing available, use entire vocabulary
        if len(available) == 0:
            available = np.arange(self.vocab_size, dtype=np.int64)
            
        if self.verbose:
            print(f'Number of available nodes: {len(available)} for the center_word {center_node}')
        # Step 5 - Sample negatives
        if self.num_negative == 0:
            negatives = np.empty( (0,), dtype=np.int64 )
        else:
            replace = len(available) < self.num_negative
            negatives = self._local_rng.choice(
                available, size=self.num_negative, replace=replace
            ).astype(np.int64)
        if self.verbose:
            print(f'Negatives: {negatives}')    
            print(f'Number of negatives: {len(negatives)}')
            print(f'Out of {self.vocab_size} nodes, {len(excluded)} are excluded.')
            print(f'Number of available nodes: {len(available)}')

            print('\nWe successfully got a training example!')
        return (
            np.int64(center_idx),
            np.int64(context_idx),
            negatives,
        )
    # ========================================================================
    # PROVIDED HELPER METHODS (no changes needed)
    # ========================================================================

    def get_sample_weights(self) -> np.ndarray:
        """
        Return per-pair weights for WeightedRandomSampler.
        
        Usage:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=dataset.get_sample_weights(),
                num_samples=len(dataset),
                replacement=True
            )
            loader = DataLoader(dataset, sampler=sampler, batch_size=32)
        """
        return self.weights

    def __len__(self) -> int:
        """Number of positive training pairs."""
        return len(self.pairs)

    def _print_stats(self):
        """Print dataset statistics for verbose."""
        print("\n📊 SkipGramDataset Statistics:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Positive pairs: {len(self.pairs):,}")
        print(f"  Negatives per positive: {self.num_negative}")
        print(f"  Total samples per epoch: {len(self.pairs) * (1 + self.num_negative):,}")
        
        if self.weights.size > 0:
            print(f"\n  Weight distribution:")
            print(f"    Min: {self.weights.min():.6f}")
            print(f"    Mean: {self.weights.mean():.6f}")
            print(f"    Median: {np.median(self.weights):.6f}")
            print(f"    Max: {self.weights.max():.6f}")
        else:
            print("  ⚠️  No pairs found - check your graph and nodes!")




# ============================================================================
# Model
# ============================================================================

"""
SkipGramModelStarter Code
=====================================

LEARNING OBJECTIVES:
1. Understand dual embedding spaces (center vs context) in Skip-Gram
2. Implement negative sampling loss with label smoothing
3. Learn proper weight initialization for embedding layers
4. Master PyTorch's batched matrix operations

WHAT YOU'LL IMPLEMENT:
- [ ] _init_embeddings(): Initialize embedding weights properly
- [ ] forward(): Compute Skip-Gram Negative Sampling (SGNS) loss
- [ ] get_embeddings(): Extract learned embeddings for downstream use

KEY CONCEPTS:
- Center embeddings: Represent words as "query" vectors
- Context embeddings: Represent words as "key" vectors  
- Why two spaces? Asymmetry helps distinguish "is context of" from "has context"
- Negative sampling: Contrastive learning - push apart unrelated pairs
"""

class SkipGramModel(nn.Module):
    """
    Skip-Gram model with Negative Sampling (SGNS).
    
    Architecture:
        - center_embeddings: Embedding(V, D) - represents words as query vectors
        - context_embeddings: Embedding(V, D) - represents words as key vectors
        - dropout: Regularization applied to center embeddings
    
    Why two embedding matrices?
        In Skip-Gram, words play two roles:
        1. As CENTER: "What contexts does this word appear in?"
        2. As CONTEXT: "What centers is this word a context for?"
        
        These are asymmetric relationships. Using separate embeddings lets the
        model learn different representations for each role, improving quality.
    
    Training objective:
        Maximize: P(context | center) for true pairs
        Minimize: P(negative | center) for random pairs
        
    Example:
        >>> model = SkipGramModel(vocab_size=1000, embedding_dim=128)
        >>> center = torch.tensor([5, 10])      # batch of 2 center words
        >>> context = torch.tensor([8, 15])     # their true contexts
        >>> negatives = torch.randint(0, 1000, (2, 10))  # 10 negatives each
        >>> loss = model(center, context, negatives)
        >>> print(loss.shape)  # torch.Size([2]) - loss per example
    """

    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float = 0.3, verbose = False):
        """
        Initialize Skip-Gram model.
        
        Args:
            vocab_size: Size of vocabulary (number of unique nodes/words)
            embedding_dim: Dimensionality of embedding vectors (typically 50-300)
            dropout: Dropout probability for regularization (prevents overfitting)
        """
        super().__init__()
        
        # Two embedding matrices: one for center words, one for context words
        # WHY: Asymmetric roles in Skip-Gram (see class docstring)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Dropout for regularization (applied only to center embeddings during training)
        # WHY: Prevents model from memorizing training pairs, improves generalization
        self.dropout = nn.Dropout(dropout)

        self.embedding_dim = embedding_dim
        self.verbose = verbose
        
        self._init_embeddings()

    def _init_embeddings(self):
        """
        Initialize embedding weights using uniform distribution.
        
        Why initialization matters:
            - Too large: Training becomes unstable (exploding gradients)
            - Too small: Learning is slow (vanishing gradients)  
            - Rule of thumb: scale inversely with embedding dimension
        
        Standard practice for Skip-Gram:
            - Use uniform distribution: U(-scale, scale)
            - Scale = 0.5 / sqrt(embedding_dim) OR 0.5 / embedding_dim
            - We use 0.5 / embedding_dim for slightly more conservative init
        """
        
        scale = 0.5/self.embedding_dim
        
        nn.init.uniform_( self.center_embeddings.weight, -scale, scale)
        nn.init.uniform_(self.context_embeddings.weight, -scale, scale)
         

    def forward(
        self, 
        center: torch.Tensor,      # shape: (batch_size,)
        context: torch.Tensor,     # shape: (batch_size,)
        negatives: torch.Tensor,   # shape: (batch_size, num_negatives)
        apply_dropout: bool = True,
        label_smoothing: float = 0.1
    ) -> torch.Tensor:
        """
        Compute Skip-Gram Negative Sampling loss.
        
        Algorithm:
        1. Look up embeddings for center, context, and negative words
        2. Compute positive score: similarity(center, context)
        3. Compute negative scores: similarity(center, each negative)
        4. Apply label smoothing to targets (anti-overfitting)
        5. Compute binary cross-entropy loss using log-sigmoid
        6. Return negative loss (we'll minimize this, which maximizes log-likelihood)
        
        Mathematical formulation:
            Positive loss: -log(σ(center · context))
            Negative loss: -Σ log(σ(-center · negative_i))
            
            With label smoothing (α = 0.1):
            - True positive target: 0.9 instead of 1.0
            - True negative target: 0.9 instead of 1.0
            This prevents overconfident predictions
        
        Args:
            center: Batch of center word indices, shape (B,)
            context: Batch of true context word indices, shape (B,)
            negatives: Batch of negative word indices, shape (B, K)
            apply_dropout: Whether to apply dropout to center embeddings
            label_smoothing: Smoothing factor (0 = no smoothing, 0.1 = mild)
            
        Returns:
            loss: Per-example loss, shape (B,). Caller typically does loss.mean()
        """
        
        center_emb = self.center_embeddings(center)          # (B, D)
        if apply_dropout:
            center_emb = self.dropout(center_emb)
            
        if self.verbose:
            print(f'\n Center embedding shape: {center_emb.shape}')
            print(f' Shape: (batch_size, embedding_dim) \n')
        
        context_emb = self.context_embeddings(context)
        if self.verbose:
            print(f' Context embedding shape: {context_emb.shape}')
            print(f' Shape: (batch_size, embedding_dim) \n')
        negative_emb = self.context_embeddings(negatives)
        if self.verbose:
            print(f' Negative embedding shape: {negative_emb.shape}')
            print(f' Shape: (batch_size, num_negatives, embedding_dim) \n')

        pos_logits = (center_emb * context_emb).sum(dim=1)   # (B,)
        if self.verbose:
            print(f' Positive loss shape: {pos_logits.shape}')
            print(f' Shape: (batch_size, )\n')
        neg_logits = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)  # (B, K)
        if self.verbose:
            print(f' Negative loss shape: {neg_logits.shape}')
            print(f' Shape: (batch_size, num_negatives)\n')
        

        # Apply label smoothing and compute losses
        # Positive loss with label smoothing:
        # ( (1-α) * log(sigmoid(pos_score)) + α * log(sigmoid(-pos_score))) 
        a= label_smoothing
        pos_loss = (  (1-a)*F.logsigmoid(pos_logits) + a *F.logsigmoid(-pos_logits)  ) # 1.0 - label_smoothing_FORMULA
        if self.verbose:
            print(f' Smoothed positive loss shape: {pos_loss.shape}')
            print(f' Shape: (batch_size,)\n')
    
        # Negative loss with label smoothing:
        # -(α * log(sigmoid(neg_score)) + (1-α) * log(sigmoid(-neg_score)))
        neg_loss = (  a*F.logsigmoid(neg_logits) + (1-a)*F.logsigmoid(-neg_logits)  )    # 1.0 - label_smoothing_FORMULA
        neg_loss = neg_loss.sum(dim=1) 
        if self.verbose:
            print(f' Smoothed negative loss shape: {neg_loss.shape}')
            print(f' Shape: (batch_size,)\n') 
        

        return -(pos_loss + neg_loss)
         

    def get_embeddings(self) -> np.ndarray:
        """
        Extract the learned center embeddings as a numpy array.
        
        Why center embeddings?
            Both center and context embeddings contain learned information, but:
            - Center embeddings are what we optimized as "query" vectors
            - They're used during training with dropout (more robust)
            - Convention: use center embeddings for downstream tasks
            
        Alternative: You could average center + context embeddings, but this
        is less common and may not improve quality.
        
        Returns:
            embeddings: numpy array of shape (vocab_size, embedding_dim)
        """
        return (
            self.center_embeddings.weight
            .detach()
            .cpu()
            .numpy()
        )



# ============================================================================
# Training
# ============================================================================

def train_embeddings(
    network_data,
    embedding_dim=128,
    batch_size=512,
    epochs=20,
    learning_rate=0.001,
    num_negative=15,
    validation_fraction=0.05,
    context_size=1,
    dropout=0.3,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=3,
    device=None,
    model_save_path="best_model.pth",
    save_plot=True
):
    """
    Train Skip-Gram embeddings with weighted sampling.
    
    Args:
        network_data: Dict with 'graph', 'nodes', 'distance_matrix'
        embedding_dim: Embedding dimensionality
        batch_size: Training batch size
        epochs: Maximum epochs
        learning_rate: Initial learning rate
        num_negative: Negatives per positive (5-20 recommended)
        validation_fraction: Fraction for validation
        context_size: Graph distance for context (1=neighbors)
        dropout: Dropout rate (default: 0.3)
        weight_decay: L2 regularization (default: 1e-4)
        label_smoothing: Label smoothing factor (default: 0.1)
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
        save_plot: Save training curve
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Filter punctuation
    network_data = filter_punctuation_from_network(network_data)
    nodes = network_data['nodes']
    graph = network_data['graph']
    distance_matrix = network_data['distance_matrix']
    # print(nodes)
    # Split edges
    all_edges = list(graph.edges())
    np.random.shuffle(all_edges)
    split_idx = int(len(all_edges) * (1 - validation_fraction))
    
    train_graph = nx.Graph()
    train_graph.add_nodes_from(nodes)
    train_graph.add_edges_from(all_edges[:split_idx])
    
    val_graph = nx.Graph()
    val_graph.add_nodes_from(nodes)
    val_graph.add_edges_from(all_edges[split_idx:])
    
    print(f"\nTrain edges: {len(all_edges[:split_idx]):,}, Val edges: {len(all_edges[split_idx:]):,}")
    print(train_graph)
    # Create datasets
    train_dataset = SkipGramDataset(train_graph, nodes, distance_matrix, num_negative, context_size)
    val_dataset = SkipGramDataset(val_graph, nodes, distance_matrix, num_negative, context_size)
    
    # Create loaders with weighted sampling
    sampler = WeightedRandomSampler(
        weights=train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Initialize model
    model = SkipGramModel(len(nodes), embedding_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\nTraining on {device}")
    print(f"Vocab: {len(nodes)}, Embed dim: {embedding_dim}, Context: {context_size}, Negatives: {num_negative}")
    print(f"Regularization: dropout={dropout}, weight_decay={weight_decay}, label_smoothing={label_smoothing}")
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for i, (centers, contexts, negs) in train_pbar:
            centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
            
            loss = model(centers, contexts, negs, True, label_smoothing).mean()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'train_loss': f'{total_loss / (i + 1):.4f}'})
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        total_val_loss = 0.0
        
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=False)
        with torch.no_grad():
            for i, (centers, contexts, negs) in val_pbar:
                centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
                
                batch_loss = model(centers, contexts, negs, False, 0.0).mean().item()
                total_val_loss += batch_loss
                val_pbar.set_postfix({'val_loss': f'{total_val_loss / (i + 1):.4f}'})
        
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  
            best_model_state = model.state_dict()          
            save_data = {
                'model_state_dict': best_model_state,
                'nodes': nodes,
                'vocab_size': len(nodes),
                'embedding_dim': embedding_dim
            }
            torch.save(save_data, model_save_path)        
            print(f"  → Best model (val_loss={best_val_loss:.4f}), saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save plot
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'o-', label='Train', linewidth=2, markersize=6)
        plt.plot(val_losses, 's-', label='Validation', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150)
        print("\nSaved loss plot to training_loss.png")
        plt.close()
    
    return {
        'nodes': nodes,
        'embeddings': model.get_embeddings(),
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


# ============================================================================
# Analysis
# ============================================================================

def find_similar_words(word, nodes, embeddings, top_k=10):
    """Find most similar words using cosine similarity."""
    if word not in nodes:
        return []
    
    idx = nodes.index(word)
    target_vec = embeddings[idx]
    
    similarities = (embeddings @ target_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
    top_indices = np.argsort(-similarities)[1:top_k+1]
    
    return [(nodes[i], float(similarities[i])) for i in top_indices]


def solve_analogy(word_a, word_b, word_c, nodes, embeddings, top_k=5):
    """Solve word analogies: word_a is to word_b as word_c is to ?"""
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    if not all(w in node_to_idx for w in [word_a, word_b, word_c]):
        return []
    
    target_vec = embeddings[node_to_idx[word_b]] - embeddings[node_to_idx[word_a]] + embeddings[node_to_idx[word_c]]
    similarities = (embeddings @ target_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
    
    exclude = {node_to_idx[w] for w in [word_a, word_b, word_c]}
    results = [(nodes[i], float(similarities[i])) for i in np.argsort(-similarities) if i not in exclude][:top_k]
    
    return results


def visualize_embeddings(nodes, embeddings, output_file="embeddings_tsne.png", 
                        sample_size=200, annotate=True):
    """Create t-SNE visualization of embeddings."""
    n_samples = min(sample_size, len(nodes))
    selected_embeddings = embeddings[:n_samples]
    selected_nodes = nodes[:n_samples]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    plt.figure(figsize=(14, 14))
    plt.scatter(projection[:, 0], projection[:, 1], s=40, alpha=0.6, c='steelblue')
    
    if annotate:
        for i, word in enumerate(selected_nodes):
            plt.annotate(word, (projection[i, 0], projection[i, 1]), fontsize=9, alpha=0.8)
    
    plt.title(f"t-SNE Visualization of Top {n_samples} Word Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE to {output_file}")
    plt.close()



def analyze_embeddings(nodes, embeddings, 
                       similarity_examples=None,
                       analogy_examples=None,
                       cluster_seeds=None):
    """Comprehensive analysis of learned embeddings."""
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    print(f"\nVocabulary: {len(nodes):,}  Embedding dim: {embeddings.shape[1]}")
    
    # Similarity statistics
    sample_emb = embeddings[:min(100, len(embeddings))]
    norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
    normalized = sample_emb / (norms + 1e-10)
    sim_matrix = normalized @ normalized.T
    sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    
    print(f"\nSimilarity stats (100 word sample):")
    print(f"  Mean: {sim_values.mean():.4f}  Std: {sim_values.std():.4f}")
    print(f"  Min: {sim_values.min():.4f}  Max: {sim_values.max():.4f}")
    
    # Nearest neighbors
    if similarity_examples:
        print("\n" + "="*80)
        print("NEAREST NEIGHBORS")
        print("="*80)
        for word in similarity_examples:
            similar = find_similar_words(word, nodes, embeddings, top_k=8)
            print(f"\nMost similar to '{word}':")
            if not similar:
                print("  (not in vocabulary)")
            else:
                for token, score in similar:
                    print(f"  {token:15s}  similarity={score:.4f}")
    
    # Analogies
    if analogy_examples:
        print("\n" + "="*80)
        print("WORD ANALOGIES (a:b :: c:?)")
        print("="*80)
        for a, b, c in analogy_examples:
            results = solve_analogy(a, b, c, nodes, embeddings, top_k=3)
            print(f"\n{a}:{b} :: {c}:?")
            if results:
                for token, score in results:
                    print(f"  {token:15s}  score={score:.4f}")
            else:
                print("  (words not in vocabulary)")
    
    # Semantic clusters
    if cluster_seeds:
        print("\n" + "="*80)
        print("SEMANTIC CLUSTERS")
        print("="*80)
        for seed in cluster_seeds:
            if seed in nodes:
                cluster = find_similar_words(seed, nodes, embeddings, top_k=5)
                print(f"\n'{seed}': {', '.join([w for w, _ in cluster])}")
    
    print("\n" + "="*80)


"""
Unit Tests for Skip-Gram with Negative Sampling

""" 


def _compute_distance_matrix_unweighted(graph: nx.Graph, nodes: List[str]) -> np.ndarray:
    """
    Build (V,V) shortest-path distance matrix aligned to `nodes`.
    Unreachable distances are np.inf. Diagonal is 0 if node exists in graph.
    """
    V = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    D = np.full((V, V), np.inf, dtype=np.float32)

    for i, u in enumerate(nodes):
        if u in graph:
            D[i, i] = 0.0
            lengths = nx.single_source_shortest_path_length(graph, u)
            for v, d in lengths.items():
                if v in node_to_idx:
                    D[i, node_to_idx[v]] = float(d)
    return D


class TestSkipGramDataset(unittest.TestCase):
    """Tests for SkipGramDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        # Graph: a-b-c-d (chain) and isolated node x
        self.graph = nx.Graph()
        self.graph.add_nodes_from(["a", "b", "c", "d", "x"])
        self.graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

        self.nodes = ["a", "b", "c", "d", "x"]
        self.distance_matrix = _compute_distance_matrix_unweighted(self.graph, self.nodes)

        # Silence dataset stats printing during tests
        with contextlib.redirect_stdout(io.StringIO()):
            self.dataset = SkipGramDataset(
                graph=self.graph,
                nodes=self.nodes,
                distance_matrix=self.distance_matrix,
                num_negative=4,
                context_size=1,
            )

    def test_initialization(self):
        """Test dataset initializes correctly."""
        self.assertIsInstance(self.dataset.graph, nx.Graph)
        self.assertEqual(self.dataset.nodes, self.nodes)
        self.assertEqual(self.dataset.vocab_size, len(self.nodes))
        self.assertEqual(self.dataset.num_negative, 4)
        self.assertEqual(self.dataset.distance_matrix.shape, (len(self.nodes), len(self.nodes)))
        self.assertIsInstance(self.dataset.node_to_idx, dict)
        self.assertIsNone(self.dataset._local_rng)

    def test_vocab_mapping(self):
        """Test node to index mapping is correct."""
        for i, n in enumerate(self.nodes):
            self.assertIn(n, self.dataset.node_to_idx)
            self.assertEqual(self.dataset.node_to_idx[n], i)

    def test_distance_matrix_shape_validation(self):
        V = len(self.nodes)
        bad = np.zeros((V + 1, V), dtype=np.float32)  # mismatched shape
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                SkipGramDataset(self.graph, self.nodes, bad, num_negative=2, context_size=1)


    def test_context_building(self):
        """Test context sets are built correctly."""
        # For context_size=1 on chain a-b-c-d
        self.assertEqual(self.dataset.contexts["a"], {"b"})
        self.assertEqual(self.dataset.contexts["b"], {"a", "c"})
        self.assertEqual(self.dataset.contexts["c"], {"b", "d"})
        self.assertEqual(self.dataset.contexts["d"], {"c"})
        # isolated node
        self.assertEqual(self.dataset.contexts["x"], set())

        # every node has a set
        for n in self.nodes:
            self.assertIn(n, self.dataset.contexts)
            self.assertIsInstance(self.dataset.contexts[n], set)
            self.assertNotIn(n, self.dataset.contexts[n])  # exclude self

    def test_context_size_parameter(self):
        """Test that context_size controls neighborhood size."""
        with contextlib.redirect_stdout(io.StringIO()):
            ds1 = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=2, context_size=1)
            ds2 = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=2, context_size=2)

        # for node "b": within 1 hop -> {a,c}; within 2 hops -> {a,c,d}
        self.assertEqual(ds1.contexts["b"], {"a", "c"})
        self.assertEqual(ds2.contexts["b"], {"a", "c", "d"})
        # for node "a": within 2 hops includes "c"
        self.assertEqual(ds2.contexts["a"], {"b", "c"})

    def test_pairs_generation(self):
        """Test that (center, context) pairs are generated."""
        self.assertIsInstance(self.dataset.pairs, list)
        self.assertGreater(len(self.dataset.pairs), 0)

        # Each pair should correspond to an actual context relationship
        for center_idx, ctx_idx in self.dataset.pairs[:20]:
            center_word = self.nodes[center_idx]
            ctx_word = self.nodes[ctx_idx]
            self.assertIn(ctx_word, self.dataset.contexts[center_word])

    def test_weights_generation(self):
        """Test that sample weights are generated correctly."""
        self.assertIsInstance(self.dataset.weights, np.ndarray)
        self.assertEqual(self.dataset.weights.dtype, np.float32)
        self.assertEqual(len(self.dataset.weights), len(self.dataset.pairs))

        if len(self.dataset.weights) > 0:
            self.assertTrue(np.isfinite(self.dataset.weights).all())
            self.assertTrue((self.dataset.weights >= 1e-6).all())
            # weights normalized ~ mean 1
            self.assertAlmostEqual(float(self.dataset.weights.mean()), 1.0, places=3)

    def test_weights_frequency_correlation(self):
        """Test that higher frequency pairs get higher weights."""
        # In this lab’s weighting, "higher frequency" is represented as "closer" (smaller distance).
        # So we craft a distance matrix where a-b is "closer" than c-d and check weight(a,b) > weight(c,d).

        nodes = ["a", "b", "c", "d"]
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("c", "d")])  # two separate edges
        D = np.full((4, 4), np.inf, dtype=np.float32)
        np.fill_diagonal(D, 0.0)

        # Both are adjacent in graph, but we "encode frequency" via distance matrix:
        # a-b very close (1), c-d less close (2)
        D[0, 1] = D[1, 0] = 1.0
        D[2, 3] = D[3, 2] = 2.0

        with contextlib.redirect_stdout(io.StringIO()):
            ds = SkipGramDataset(g, nodes, D, num_negative=2, context_size=1)

        # Find weights for (a,b) and (c,d)
        pair_to_w = {p: float(w) for p, w in zip(ds.pairs, ds.weights)}
        a_idx, b_idx = ds.node_to_idx["a"], ds.node_to_idx["b"]
        c_idx, d_idx = ds.node_to_idx["c"], ds.node_to_idx["d"]

        wab = pair_to_w.get((a_idx, b_idx), None)
        wcd = pair_to_w.get((c_idx, d_idx), None)

        # contexts are symmetric in an undirected graph, so pairs should exist
        self.assertIsNotNone(wab)
        self.assertIsNotNone(wcd)
        self.assertGreater(wab, wcd)

    def test_dataset_length(self):
        """Test that dataset length equals number of pairs."""
        self.assertEqual(len(self.dataset), len(self.dataset.pairs))

    def test_empty_graph_handling(self):
        """Test behavior with disconnected nodes."""
        g = nx.Graph()
        g.add_nodes_from(["a", "b", "c"])
        nodes = ["a", "b", "c"]
        D = _compute_distance_matrix_unweighted(g, nodes)

        with contextlib.redirect_stdout(io.StringIO()):
            ds = SkipGramDataset(g, nodes, D, num_negative=2, context_size=1)

        # no edges => no contexts => no pairs
        self.assertEqual(len(ds), 0)
        for n in nodes:
            self.assertEqual(ds.contexts[n], set())

    def test_getitem_structure(self):
        """Test that getitem returns correct structure."""
        if len(self.dataset) == 0:
            self.skipTest("No pairs in dataset; cannot test getitem")

        center_idx, context_idx, negatives = self.dataset[0]
        self.assertIsInstance(center_idx, np.int64)
        self.assertIsInstance(context_idx, np.int64)
        self.assertIsInstance(negatives, np.ndarray)
        self.assertEqual(negatives.dtype, np.int64)
        self.assertEqual(tuple(negatives.shape), (self.dataset.num_negative,))

    def test_negative_sampling_excludes_context(self):
        """Test that negative samples exclude center and its context."""
        if len(self.dataset) == 0:
            self.skipTest("No pairs in dataset; cannot test negative sampling")

        center_idx, context_idx, negatives = self.dataset[0]
        center_word = self.nodes[int(center_idx)]
        excluded_words = self.dataset.contexts[center_word] | {center_word}
        excluded = {self.dataset.node_to_idx[w] for w in excluded_words}

        for n in negatives.tolist():
            self.assertNotIn(int(n), excluded)

        # context should be true context
        ctx_word = self.nodes[int(context_idx)]
        self.assertIn(ctx_word, self.dataset.contexts[center_word])

    def test_negative_sampling_consistency(self):
        """Test that negative sampling produces valid samples consistently."""
        if len(self.dataset) == 0:
            self.skipTest("No pairs in dataset; cannot test negative sampling")

        for i in range(min(10, len(self.dataset))):
            center_idx, context_idx, negatives = self.dataset[i]
            self.assertEqual(tuple(negatives.shape), (self.dataset.num_negative,))
            self.assertTrue((negatives >= 0).all())
            self.assertTrue((negatives < self.dataset.vocab_size).all())


class TestSkipGramModel(unittest.TestCase):
    """Tests for SkipGramModel class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(0)
        self.vocab_size = 10
        self.embedding_dim = 8
        self.model = SkipGramModel(self.vocab_size, self.embedding_dim, dropout=0.5, verbose=False)

    def test_initialization(self):
        """Test model initializes correctly."""
        self.assertIsInstance(self.model.center_embeddings, nn.Embedding)
        self.assertIsInstance(self.model.context_embeddings, nn.Embedding)
        self.assertIsInstance(self.model.dropout, nn.Dropout)

    def test_embedding_dimensions(self):
        """Test embedding layers have correct dimensions."""
        self.assertEqual(self.model.center_embeddings.weight.shape, (self.vocab_size, self.embedding_dim))
        self.assertEqual(self.model.context_embeddings.weight.shape, (self.vocab_size, self.embedding_dim))

    def test_different_center_context_embeddings(self):
        """Test that center and context embeddings are separate."""
        self.assertIsNot(self.model.center_embeddings.weight, self.model.context_embeddings.weight)

    def test_embedding_initialization_range(self):
        """Test that embeddings are initialized in reasonable range."""
        scale = 0.5 / self.embedding_dim
        w = self.model.center_embeddings.weight.detach().cpu()
        self.assertTrue(torch.isfinite(w).all())
        self.assertLessEqual(float(w.max()), scale + 1e-6)
        self.assertGreaterEqual(float(w.min()), -scale - 1e-6)

    def test_forward_pass_shape(self):
        """Test forward pass returns correct shape."""
        B, K = 4, 5
        center = torch.randint(0, self.vocab_size, (B,))
        context = torch.randint(0, self.vocab_size, (B,))
        negatives = torch.randint(0, self.vocab_size, (B, K))

        loss = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0)
        self.assertEqual(tuple(loss.shape), (B,))

    def test_forward_pass_no_nan(self):
        """Test forward pass doesn't produce NaN values."""
        B, K = 8, 7
        center = torch.randint(0, self.vocab_size, (B,))
        context = torch.randint(0, self.vocab_size, (B,))
        negatives = torch.randint(0, self.vocab_size, (B, K))

        loss = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0)
        self.assertTrue(torch.isfinite(loss).all())

    def test_loss_is_positive(self):
        """Test that loss values are positive (negative log likelihood)."""
        B, K = 6, 4
        center = torch.randint(0, self.vocab_size, (B,))
        context = torch.randint(0, self.vocab_size, (B,))
        negatives = torch.randint(0, self.vocab_size, (B, K))

        loss = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0)
        self.assertTrue((loss > 0).all())

    def test_label_smoothing_effect(self):
        """Test that label smoothing affects loss."""
        B, K = 6, 4
        center = torch.randint(0, self.vocab_size, (B,))
        context = torch.randint(0, self.vocab_size, (B,))
        negatives = torch.randint(0, self.vocab_size, (B, K))

        loss0 = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0).mean()
        loss1 = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.2).mean()
        self.assertNotAlmostEqual(float(loss0.item()), float(loss1.item()), places=6)

    def test_dropout_training_vs_eval(self):
        """Test dropout behaves differently in train vs eval mode."""
        B, K = 4, 3
        center = torch.randint(0, self.vocab_size, (B,))
        context = torch.randint(0, self.vocab_size, (B,))
        negatives = torch.randint(0, self.vocab_size, (B, K))

        torch.manual_seed(123)
        self.model.train()
        out_train = self.model(center, context, negatives, apply_dropout=True, label_smoothing=0.0)

        torch.manual_seed(123)
        self.model.eval()
        out_eval = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0)

        # Not guaranteed to differ elementwise, but should differ in aggregate for nontrivial dropout.
        self.assertNotAlmostEqual(float(out_train.mean().item()), float(out_eval.mean().item()), places=6)

    def test_batch_size_invariance(self):
        """Test model works with different batch sizes."""
        for B in [1, 2, 16]:
            K = 5
            center = torch.randint(0, self.vocab_size, (B,))
            context = torch.randint(0, self.vocab_size, (B,))
            negatives = torch.randint(0, self.vocab_size, (B, K))
            loss = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0)
            self.assertEqual(tuple(loss.shape), (B,))

    def test_negative_sample_size_invariance(self):
        """Test model works with different numbers of negative samples."""
        B = 4
        for K in [1, 2, 10]:
            center = torch.randint(0, self.vocab_size, (B,))
            context = torch.randint(0, self.vocab_size, (B,))
            negatives = torch.randint(0, self.vocab_size, (B, K))
            loss = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0)
            self.assertEqual(tuple(loss.shape), (B,))

    def test_backward_pass(self):
        """Test that gradients flow correctly through the model."""
        B, K = 8, 5
        center = torch.randint(0, self.vocab_size, (B,))
        context = torch.randint(0, self.vocab_size, (B,))
        negatives = torch.randint(0, self.vocab_size, (B, K))

        loss = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0).mean()
        loss.backward()

        self.assertIsNotNone(self.model.center_embeddings.weight.grad)
        self.assertIsNotNone(self.model.context_embeddings.weight.grad)
        self.assertTrue(torch.isfinite(self.model.center_embeddings.weight.grad).all())
        self.assertTrue(torch.isfinite(self.model.context_embeddings.weight.grad).all())

    def test_embedding_update_during_training(self):
        """Test that embeddings actually change during training."""
        B, K = 16, 5
        center = torch.randint(0, self.vocab_size, (B,))
        context = torch.randint(0, self.vocab_size, (B,))
        negatives = torch.randint(0, self.vocab_size, (B, K))

        opt = torch.optim.SGD(self.model.parameters(), lr=0.1)
        before = self.model.center_embeddings.weight.detach().clone()

        loss = self.model(center, context, negatives, apply_dropout=False, label_smoothing=0.0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        after = self.model.center_embeddings.weight.detach()
        self.assertGreater(float((after - before).abs().sum().item()), 0.0)
 

    def test_get_embeddings(self):
        """Test embedding extraction."""
        emb = self.model.get_embeddings()
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (self.vocab_size, self.embedding_dim))
        self.assertTrue(np.isfinite(emb).all())


class TestIntegration(unittest.TestCase):
    """Integration tests for dataset and model working together."""

    def setUp(self):
        """Set up integration test fixtures."""
        torch.manual_seed(0)
        np.random.seed(0)

        g = nx.Graph()
        g.add_nodes_from(["a", "b", "c", "d"])
        g.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

        self.nodes = ["a", "b", "c", "d"]
        self.D = _compute_distance_matrix_unweighted(g, self.nodes)

        with contextlib.redirect_stdout(io.StringIO()):
            self.dataset = SkipGramDataset(g, self.nodes, self.D, num_negative=3, context_size=1)

        self.model = SkipGramModel(vocab_size=len(self.nodes), embedding_dim=16, dropout=0.2, verbose=False)

    def test_dataset_model_compatibility(self):
        """Test that dataset output works with model input."""
        if len(self.dataset) == 0:
            self.skipTest("No pairs produced; cannot test compatibility")

        loader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        centers, contexts, negs = next(iter(loader))

        out = self.model(centers, contexts, negs, apply_dropout=False, label_smoothing=0.0)
        self.assertEqual(tuple(out.shape), (centers.shape[0],))
        self.assertTrue(torch.isfinite(out).all())

    def test_full_training_step(self):
        """Test a complete training step with real data."""
        if len(self.dataset) == 0:
            self.skipTest("No pairs produced; cannot test training step")

        loader = DataLoader(self.dataset, batch_size=8, shuffle=True)
        centers, contexts, negs = next(iter(loader))

        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        self.model.train()
        loss = self.model(centers, contexts, negs, apply_dropout=True, label_smoothing=0.1).mean()
        self.assertTrue(torch.isfinite(loss).all())

        before = self.model.center_embeddings.weight.detach().clone()

        opt.zero_grad()
        loss.backward()
        opt.step()

        after = self.model.center_embeddings.weight.detach()
        self.assertGreater(float((after - before).abs().sum().item()), 0.0)
 

def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING SKIP-GRAM UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramModel))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Total tests run: {result.testsRun}")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
