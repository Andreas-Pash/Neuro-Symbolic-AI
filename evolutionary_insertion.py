"""
Simple Genetic Algorithm for Word Embedding Insertion
======================================================================
A (1+λ) Evolution Strategy for inserting new word embeddings into a trained
Skip-Gram model while preserving the existing embedding space structure.
"""

import torch
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision

from skipgram_modelling import SkipGramModel, find_similar_words
from network_processing import process_text_network

import unittest
import tempfile
import os


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================
 

def load_trained_model(model_path: str, vocab_size: int, 
                       embedding_dim: int, dropout: float) -> Tuple[torch.nn.Module, np.ndarray]:
    """Load trained Skip-Gram model and extract embeddings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
     
    try:
        model = SkipGramModel(vocab_size= checkpoint['vocab_size'],
                              embedding_dim = checkpoint['embedding_dim'],
                              dropout=dropout
                            ).to(device)
        embeds_dim = checkpoint['embedding_dim']
    except:
        model = SkipGramModel(vocab_size= checkpoint['metadata']['vocab_size'], 
                              embedding_dim = checkpoint['metadata']['embedding_dim'], 
                              dropout=dropout
                            ).to(device)
        embeds_dim = checkpoint['metadata']['embedding_dim']
    
    try:
        print(f'Didnt find embeddings in checkpoint, using model.get_embeddings()')

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            embeddings_tensor = model.get_embeddings()
            embeddings = (embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor) 
                        else embeddings_tensor).astype(np.float32)
    except:
        embeddings_tensor = checkpoint['embeddings']
        embeddings = (embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor) 
                        else embeddings_tensor).astype(np.float32)
    
    
    print(f"✓ Loaded model: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return model, embeddings


def create_mappings(nodes: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, np.ndarray]]:
    """Create word-to-index and index-to-word mappings."""
    word_to_idx = {word: idx for idx, word in enumerate(nodes)}
    idx_to_word = {idx: word for idx, word in enumerate(nodes)}
    return word_to_idx, idx_to_word


def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute statistics needed for fitness evaluation."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'global_std': np.std(embeddings)
    }


def get_cifar100_vocabulary() -> List[str]:
    """Download CIFAR-100 and extract class names."""
    print("\nLoading CIFAR-100 vocabulary...")
    dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True)
    print(f"✓ CIFAR-100 vocabulary loaded: {len(dataset.classes)} classes")
    return dataset.classes


def analyze_vocabulary_overlap(cifar_vocab: List[str], network_vocab: List[str]) -> List[str]:
    """Analyze overlap between CIFAR-100 and network vocabulary."""
    cifar_set, network_set = set(cifar_vocab), set(network_vocab)
    overlapping = sorted(list(cifar_set.intersection(network_set)))
    missing = sorted(list(cifar_set - network_set))
    
    print(f"\n{'='*70}")
    print("VOCABULARY OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"CIFAR-100 vocabulary: {len(cifar_set)} classes")
    print(f"Network vocabulary: {len(network_set)} words")
    print(f"Overlapping words: {len(overlapping)} ({len(overlapping)/len(cifar_set)*100:.1f}%)")
    print(f"Missing from network: {len(missing)}")
    if overlapping:
        print(f"\nFound: {', '.join(overlapping)}")
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
    print(f"{'='*70}\n")
    
    return missing


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

def extract_word_contexts(
    text_file: str,
    target_words: List[str],
    vocab_set: Set[str],
    window: int = 5
) -> Dict[str, Counter]:
    """
    Extract co-occurrence context statistics for target words from a text corpus.
    
    This function reads a corpus file line-by-line and tracks which words appear
    near specified target words. For each target word, it counts how many times
    each vocabulary word appears within a window around it.
    
    Args:
        text_file: Path to the corpus text file to analyze.
        target_words: List of words to extract contexts for.
        vocab_set: Set of valid vocabulary words (only count these as contexts).
        window: Number of words to look on each side of the target word.
    
    Returns:
        A dictionary mapping each target word to a Counter of context words and
        their frequencies.
        
    Example:
        >>> extract_word_contexts('corpus.txt', ['king', 'queen'], vocab, window=2)
        {'king': Counter({'royal': 5, 'crown': 3}), 
         'queen': Counter({'royal': 4, 'throne': 2})}
    
    Implementation guidelines:
    --------------------------
    1. Initialize a dictionary `{word: Counter()}` for each target word.
    2. Convert `target_words` to a set for fast lookup.
    3. Stream through the file line-by-line (efficient for large corpora).
    4. For each line:
        - Tokenize using lowercase alphabetic words (regex: r"\\b[a-z]+\\b").
        - For each token that matches a target word:
            * Extract up to `window` tokens on both sides.
            * Exclude the target word itself.
            * Retain only context words that appear in `vocab_set`.
            * Update the Counter for that target word.
    5. Handle edge cases: empty lines, start/end of token lists.
    6. Optionally print progress (e.g., every 50,000 lines) for user feedback.
    7. Return the dictionary of Counters.
    """
     
    contexts = {word: Counter() for word in target_words}
    target_set = set(target_words)
    vocab_set = set(vocab_set)

    with open(text_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                print(f"Processed {i} lines...")
            tokens = re.findall(r"\b[a-z]+\b", line.lower())
            for j, token in enumerate(tokens):
                if token in target_set:
                    left_context = tokens[max(0, j - window):j]
                    right_context = tokens[j + 1:min(j + window + 1, len(tokens))]
                    context_words = [word for word in left_context + right_context if word in vocab_set and word != token]
                    contexts[token].update(context_words) 
    print(f"Completed. context stats:")
    for word in target_words:
        # print(f"Contexts for '{word}': {contexts[word]}")
        print(f"""  {word:10s}: {sum(contexts[word].values()):6d} contexts, {len(contexts[word]):3d} unique words""")
    
    return contexts






# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_fitness(
    vec: np.ndarray,
    word: str,
    ctx_vecs: Optional[np.ndarray],
    ctx_weights: Optional[np.ndarray],
    neg_vecs: np.ndarray,
    anchor_vecs: Optional[np.ndarray],
    stats_dict: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute a three-term fitness score for a candidate word embedding vector.
    
    This function evaluates how well a candidate vector fits the learned 
    embedding space by combining three complementary metrics:
    1. Corpus likelihood (how well it predicts observed contexts)
    2. Norm matching (how similar its magnitude is to typical embeddings)
    3. Anchor similarity (how similar it is to known reference words)
    
    Args:
        vec: Candidate embedding vector to evaluate.
        word: Target word (for reference, not used in computation).
        ctx_vecs: Context word vectors that co-occur with the target word.
                  Shape: (n_contexts, embedding_dim). May be None if no contexts.
        ctx_weights: Weights for each context (e.g., co-occurrence counts).
                     Shape: (n_contexts,). May be None if no contexts.
        neg_vecs: Negative sample vectors (words that don't co-occur).
                  Shape: (n_negatives, embedding_dim).
        anchor_vecs: Pre-normalized vectors of anchor words for comparison.
                     Shape: (n_anchors, embedding_dim). May be None.
        stats_dict: Dictionary containing embedding statistics:
                    - 'mean_norm': Average L2 norm of embeddings in the space
                    - 'std_norm': Standard deviation of embedding norms
                    - 'global_std': Global standard deviation (if needed)
        weights: Dictionary of weights for each fitness component:
                 - 'corpus': Weight for corpus likelihood term
                 - 'norm': Weight for norm matching term
                 - 'anchor': Weight for anchor similarity term
    
    Returns:
        Combined fitness score in the range [0, 1], where higher is better.
        
    Example:
        >>> vec = np.array([0.5, -0.3, 0.8, 0.1])
        >>> stats = {'mean_norm': 1.0, 'std_norm': 0.2, 'global_std': 0.5}
        >>> weights = {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
        >>> fitness = compute_fitness(vec, 'king', ctx_vecs, ctx_weights, 
        ...                           neg_vecs, anchor_vecs, stats, weights)
        >>> print(f"Fitness: {fitness:.4f}")
        Fitness: 0.7234
    
    Implementation guidelines:
    --------------------------
    Term 1 - Corpus Likelihood (L_corpus_norm):
        - For positive contexts: sum over ctx_weights * log(sigmoid(ctx_vecs · vec))
        - For negative samples: sum over log(sigmoid(-neg_vecs · vec))
        - Add small epsilon (1e-10) inside log for numerical stability
        - Normalize by total samples, then apply sigmoid to map to [0, 1]
        - Default to 0.5 if no samples available
        
    Term 2 - Norm Match (S_norm):
        - Compute L2 norm of the candidate vector
        - Use Gaussian similarity: exp(-((norm - mean_norm)² / (2 * std_norm²)))
        - This rewards vectors with norms close to the typical embedding norm
        
    Term 3 - Anchor Similarity (S_anchor):
        - Normalize the candidate vector (divide by its norm + epsilon)
        - Compute dot products with all anchor vectors (they're pre-normalized)
        - Take the mean similarity across all anchors
        - Default to 0.5 if no anchors provided
        
    Final score:
        - Weighted sum: weights['corpus'] * L_corpus_norm + 
                       weights['norm'] * S_norm + 
                       weights['anchor'] * S_anchor
    
    Notes:
        - Handle None values for optional parameters (ctx_vecs, ctx_weights, anchor_vecs)
        - Use vectorized NumPy operations for efficiency
        - Add small epsilon values to prevent division by zero
    """
    
    eps = 1e-10
    v = np.asarray(vec, dtype=np.float64)

    # -------------------------
    # Term 1: Corpus likelihood
    # -------------------------
    terms = []

    # Positive contexts (weighted)
    if ctx_vecs is not None and ctx_weights is not None and len(ctx_vecs) > 0:
        cv = np.asarray(ctx_vecs, dtype=np.float64)
        w = np.asarray(ctx_weights, dtype=np.float64).reshape(-1)

        if w.size == cv.shape[0] and w.sum() > 0:
            # Normalize here so caller doesn't have to pre-normalize
            w = w / (w.sum() + eps)
            pos_scores = cv @ v
            pos_ll = np.sum(w * np.log(sigmoid(pos_scores) + eps))
            terms.append(pos_ll)

    # Negative samples (mean)
    if neg_vecs is not None and np.asarray(neg_vecs).size > 0 and np.asarray(neg_vecs).shape[0] > 0:
        nv = np.asarray(neg_vecs, dtype=np.float64)
        neg_scores = nv @ v
        neg_ll = np.mean(np.log(sigmoid(-neg_scores) + eps))
        terms.append(neg_ll)

    # Map to [0, 1]
    if terms:
        corpus_term = float(sigmoid(np.mean(terms)))
    else:
        corpus_term = 0.5

    # -------------------------
    # Term 2: Norm match
    # -------------------------
    mean_norm = float(stats_dict.get("mean_norm", 1.0))
    std_norm = float(stats_dict.get("std_norm", 1.0))
    norm_v = float(np.linalg.norm(v))

    if std_norm <= eps:
        norm_term = 1.0 if abs(norm_v - mean_norm) <= 1e-6 else 0.0
    else:
        norm_term = float(np.exp(-((norm_v - mean_norm) ** 2) / (2.0 * (std_norm ** 2) + eps)))

    # -------------------------
    # Term 3: Anchor similarity
    # -------------------------
    if anchor_vecs is not None and np.asarray(anchor_vecs).size > 0 and np.asarray(anchor_vecs).shape[0] > 0:
        av = np.asarray(anchor_vecs, dtype=np.float64)
        if norm_v <= eps:
            anchor_term = 0.5
        else:
            vhat = v / (norm_v + eps)
            sim = av @ vhat  # anchors are pre-normalized
            anchor_mean = float(np.mean(sim))
            anchor_term = float(np.clip((anchor_mean + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] -> [0,1]
    else:
        anchor_term = 0.5

    # -------------------------
    # Combine (keep in [0, 1])
    # -------------------------
    w_c = float(weights.get("corpus", 0.0))
    w_n = float(weights.get("norm", 0.0))
    w_a = float(weights.get("anchor", 0.0))
    w_sum = w_c + w_n + w_a

    if w_sum <= eps:
        score = 0.0
    else:
        score = (w_c * corpus_term + w_n * norm_term + w_a * anchor_term) / w_sum

    return float(np.clip(score, 0.0, 1.0))

 

# ============================================================================
# GENETIC ALGORITHM (1+λ) EVOLUTION STRATEGY
# ============================================================================

def initialize_embedding(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int]
) -> np.ndarray:
    """
    Initialize an embedding vector for a word using corpus bootstrap.
    
    This function creates an initial embedding by computing a weighted average
    of the embeddings of words that frequently co-occur with the target word.
    This provides a data-driven starting point that places the new word near
    semantically related words in the embedding space.
    
    Args:
        word: Target word to initialize an embedding for.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
    
    Returns:
        Initial embedding vector for the word. Shape: (embedding_dim,).
        
    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30, 'castle': 20})}
        >>> embeddings = np.random.randn(1000, 300)  # 1000 words, 300 dims
        >>> word_to_idx = {'queen': 0, 'royal': 1, 'castle': 2, ...}
        >>> vec = initialize_embedding('king', contexts, embeddings, word_to_idx)
        >>> vec.shape
        (300,)
    
    Implementation guidelines:
    --------------------------
    1. Handle the no-context case:
       - If the word has no contexts (empty Counter), return the mean of all
         embeddings as a neutral starting point
    
    2. Get top context words:
       - Extract the top 20 most frequent context words using Counter.most_common()
       - This focuses on the strongest statistical relationships
    
    3. Compute weighted average:
       - Calculate the total weight (sum of all counts)
       - For each context word that exists in word_to_idx:
           * Get its embedding vector
           * Weight it by (count / weight_sum)
           * Add to running sum
    
    4. Validate the result:
       - Check if the resulting vector has non-zero norm
       - If zero (e.g., no valid context words found), fall back to mean embedding
    
    Notes:
        - Some context words may not be in word_to_idx; skip these
        - The weighted average naturally places the new word near its contexts
        - Using top 20 contexts balances informativeness with noise reduction
    """
   
    mean_vec = embeddings.mean(axis=0).astype(np.float32)

    ctx_counter = contexts.get(word, Counter())
    if not ctx_counter:
        return mean_vec.copy()

    top = ctx_counter.most_common(20)

    # Keep only valid context words
    valid = [(w, c) for (w, c) in top if w in word_to_idx and c > 0]
    if not valid:
        return mean_vec.copy()

    denom = float(sum(c for _, c in valid))
    if denom <= 0:
        return mean_vec.copy()

    vec = np.zeros(embeddings.shape[1], dtype=np.float64)
    for w, c in valid:
        vec += embeddings[word_to_idx[w]].astype(np.float64) * (float(c) / denom)

    # Safety fallback
    if (not np.isfinite(vec).all()) or (np.linalg.norm(vec) <= 1e-12):
        return mean_vec.copy()

    return vec.astype(np.float32)



def precompute_fitness_vectors(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    vocab_list: List[str],
    anchors: Dict[str, List[str]],
    num_negatives: int = 15
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Precompute all vectors needed for fitness evaluation.
    
    This function extracts and prepares the three types of vectors used in
    fitness computation: positive context vectors, negative sample vectors,
    and anchor vectors. Precomputing these vectors once improves efficiency
    when evaluating fitness multiple times during optimization.
    
    Args:
        word: Target word being optimized.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
        vocab_list: List of all vocabulary words (for negative sampling).
        anchors: Dictionary mapping words to lists of semantically related anchor words.
        num_negatives: Number of negative samples to draw (default: 15).
    
    Returns:
        Tuple of (ctx_vecs, ctx_weights, neg_vecs, anchor_vecs):
        - ctx_vecs: Context word embeddings. Shape: (n_contexts, dim) or None.
        - ctx_weights: Normalized context weights. Shape: (n_contexts,) or None.
        - neg_vecs: Negative sample embeddings. Shape: (num_negatives, dim).
        - anchor_vecs: Normalized anchor embeddings. Shape: (n_anchors, dim) or None.
        
    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30})}
        >>> anchors = {'king': ['queen', 'monarch', 'ruler']}
        >>> ctx_v, ctx_w, neg_v, anc_v = precompute_fitness_vectors(
        ...     'king', contexts, embeddings, word_to_idx, vocab_list, anchors
        ... )
        >>> ctx_v.shape  # Positive contexts
        (2, 300)
        >>> neg_v.shape  # Negative samples
        (15, 300)
    
    Implementation guidelines:
    --------------------------
    Part 1 - Positive Context Vectors:
        - Initialize ctx_vecs and ctx_weights to None (for no-context case)
        - If the word has contexts:
            * Iterate through contexts[word].items()
            * For each context word that exists in word_to_idx:
              - Collect its embedding vector
              - Collect its count
            * If any valid contexts found:
              - Convert lists to numpy arrays
              - Normalize weights to sum to 1.0
    
    Part 2 - Negative Sample Vectors:
        - Randomly sample num_negatives words from vocab_list (without replacement)
        - Look up their embeddings and stack into an array
        - Shape should be (num_negatives, embedding_dim)
    
    Part 3 - Anchor Vectors:
        - Initialize anchor_vecs to None (for no-anchor case)
        - If the word has anchors defined:
            * Filter to only anchors that exist in word_to_idx
            * If any valid anchors found:
              - Collect their embeddings into an array
              - Normalize each vector to unit length (L2 norm = 1)
              - Use np.linalg.norm with axis=1, keepdims=True
              - Add small epsilon (1e-10) to prevent division by zero
    
    Notes:
        - Handle missing words gracefully (skip if not in word_to_idx)
        - Return None for optional components if no valid data available
        - Negative samples should be random to avoid bias
        - Anchor normalization enables direct cosine similarity via dot product
    """
 
    # -------------------------
    # Positive context vectors
    # -------------------------
    ctx_vecs: Optional[np.ndarray] = None
    ctx_weights: Optional[np.ndarray] = None

    counter = contexts.get(word, Counter())
    if counter:
        vec_list = []
        w_list = []
        for cw, cnt in counter.items():
            if cw in word_to_idx and cnt > 0:
                vec_list.append(embeddings[word_to_idx[cw]])
                w_list.append(float(cnt))

        if vec_list:
            ctx_vecs = np.stack(vec_list, axis=0).astype(np.float32)
            w = np.asarray(w_list, dtype=np.float64)
            w = w / (w.sum() + 1e-10)
            ctx_weights = w.astype(np.float32)

    # -------------------------
    # Negative sample vectors
    # -------------------------
    if len(vocab_list) == 0:
        neg_vecs = np.empty((0, embeddings.shape[1]), dtype=np.float32)
    else:
        replace = len(vocab_list) < num_negatives
        idxs = np.random.choice(len(vocab_list), size=num_negatives, replace=replace)
        sampled_words = [vocab_list[i] for i in idxs]

        sampled_idxs = [word_to_idx[w] for w in sampled_words if w in word_to_idx]

        # If anything got filtered out, pad by resampling
        while len(sampled_idxs) < num_negatives:
            w = vocab_list[int(np.random.randint(0, len(vocab_list)))]
            if w in word_to_idx:
                sampled_idxs.append(word_to_idx[w])

        neg_vecs = embeddings[np.asarray(sampled_idxs[:num_negatives])].astype(np.float32)

    # -------------------------
    # Anchor vectors (normalized)
    # -------------------------
    anchor_vecs: Optional[np.ndarray] = None
    anchor_words = anchors.get(word, None)

    if anchor_words:
        valid_anchors = [a for a in anchor_words if a in word_to_idx]
        if valid_anchors:
            av = np.stack([embeddings[word_to_idx[a]] for a in valid_anchors], axis=0).astype(np.float32)
            norms = np.linalg.norm(av, axis=1, keepdims=True).astype(np.float32)
            anchor_vecs = av / (norms + 1e-10)

    return ctx_vecs, ctx_weights, neg_vecs, anchor_vecs
 

def evolve_embedding(word: str, contexts: Dict[str, Counter], 
                    embeddings: np.ndarray, word_to_idx: Dict[str, int],
                    vocab_list: List[str], stats_dict: Dict[str, float],
                    anchors: Dict[str, List[str]], config: Dict) -> np.ndarray:
    """
    Evolve a single word embedding using (1+λ) Evolution Strategy.
    
    Args:
        word: Target word to insert
        contexts: Context word counts for all target words
        embeddings: Existing embedding matrix
        word_to_idx: Word to index mapping
        vocab_list: List of vocabulary words
        stats_dict: Embedding statistics
        anchors: Anchor words for semantic guidance
        config: Configuration dictionary
    
    Returns:
        Optimized embedding vector
    """
    print(f"\n  Evolving: '{word}'", end='')
    
    dim = embeddings.shape[1]
    mutation_sigma = config['ga_mutation_factor'] * stats_dict['global_std']
    
    # Initialize
    best_vec = initialize_embedding(word, contexts, embeddings, word_to_idx)
    
    # Precompute vectors
    ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
        word, contexts, embeddings, word_to_idx, vocab_list, anchors
    )
    
    # Initial fitness
    best_fit = compute_fitness(best_vec, word, ctx_vecs, ctx_weights, neg_vecs, 
                               anchor_vecs, stats_dict, config['fitness_weights'])
    
    # Evolution loop
    for gen in range(config['ga_generations']):
        # Generate offspring and evaluate
        population = best_vec + np.random.normal(0, mutation_sigma, (config['ga_pop_size'], dim))
        all_candidates = np.vstack([best_vec, population])
        
        fitness_scores = [compute_fitness(vec, word, ctx_vecs, ctx_weights, neg_vecs, 
                                         anchor_vecs, stats_dict, config['fitness_weights'])
                         for vec in all_candidates]
        
        # Select best
        best_idx = np.argmax(fitness_scores)
        best_vec = all_candidates[best_idx].copy()
        best_fit = fitness_scores[best_idx]
        
        if gen % 50 == 0:
            print(f" G{gen}={best_fit:.4f}", end='')
    
    print(f" ✓ Final={best_fit:.4f}")
    return best_vec


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_with_inserted_words(nodes: List[str], embeddings: np.ndarray, 
                                  inserted_words: List[str],
                                  output_file: str = "embeddings_with_inserted.png",
                                  sample_size: int = 500):
    """Create t-SNE visualization highlighting inserted words."""
    print("\nGenerating t-SNE visualization with inserted words...")
    
    num_original = len(nodes) - len(inserted_words)
    inserted_indices = set(range(num_original, len(nodes)))
    
    # Sample: prioritize inserted words + random original
    if len(nodes) > sample_size:
        sample_indices = list(inserted_indices) + list(np.random.choice(
            num_original, min(sample_size - len(inserted_words), num_original), replace=False))
    else:
        sample_indices = list(range(len(nodes)))
    
    selected_embeddings = embeddings[sample_indices]
    selected_nodes = [nodes[i] for i in sample_indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_indices)-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    # Plot
    plt.figure(figsize=(14, 14))
    
    for i in range(len(projection)):
        is_inserted = sample_indices[i] in inserted_indices
        plt.scatter(projection[i, 0], projection[i, 1], 
                   s=200 if is_inserted else 40,
                   alpha=1.0 if is_inserted else 0.6,
                   c='red' if is_inserted else 'steelblue')
        plt.annotate(selected_nodes[i], (projection[i, 0], projection[i, 1]), 
                    fontsize=11 if is_inserted else 9,
                    alpha=1.0 if is_inserted else 0.8,
                    fontweight='bold' if is_inserted else 'normal')
    
    plt.title(f"t-SNE Visualization: {len(sample_indices)} Words "
              f"({sum(1 for i in sample_indices if i in inserted_indices)} Inserted)",
              fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved t-SNE to {output_file}")
    plt.show()


def run_sanity_checks(model: torch.nn.Module, embeddings: np.ndarray, 
                     nodes: List[str], word_to_idx: Dict[str, int]):
    """Run comprehensive sanity checks on loaded model and embeddings."""
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    
    print(f"\n1. Model Configuration:")
    print(f"   Training mode: {model.training}")
    print(f"   Device: {next(model.parameters()).device}")
    
    print(f"\n2. Embedding Quality:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}")
    print(f"   Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}")
    print(f"   Contains NaN: {np.isnan(embeddings).any()}, Contains Inf: {np.isinf(embeddings).any()}")
    
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n3. Embedding Norms:")
    print(f"   Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
    print(f"   Range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    print(f"\n4. Vocabulary Test:")
    for test_word in ['man', 'woman', 'dog', 'car', 'blue']:
        if test_word in word_to_idx:
            word_idx = word_to_idx[test_word]
            print(f"   '{test_word:10s}' → idx={word_idx:4d}, norm={np.linalg.norm(embeddings[word_idx]):.4f}")
            similar = find_similar_words(test_word, nodes, embeddings, top_k=5)
            if similar:
                print(f"      Similar: {', '.join([f'{w}({s:.3f})' for w, s in similar])}")
    
    print("\n" + "="*70)
    print("✓ SANITY CHECKS COMPLETE")
    print("="*70)



"""
Don't forget to add tests!!!!
"""

# =============================================================================
# UNIT TESTS (Lab 7) 
# =============================================================================

import io
import contextlib

class TestExtractWordContexts(unittest.TestCase):
    """Unit tests for extract_word_contexts()."""

    def _write_tmp_corpus(self, text: str) -> str:
        fd, path = tempfile.mkstemp(prefix="lab7_corpus_", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_basic_context_extraction(self):
        """Test basic context extraction with window=2."""
        corpus = "the quick brown fox jumps over the lazy dog\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["fox"]
        vocab_set = set(re.findall(r"\b[a-z]+\b", corpus.lower()))

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=2)

        self.assertIn("fox", ctx)
        self.assertIsInstance(ctx["fox"], Counter)
        expected = Counter({"quick": 1, "brown": 1, "jumps": 1, "over": 1})
        self.assertEqual(ctx["fox"], expected)

    def test_case_insensitivity(self):
        """Test that extraction is case-insensitive."""
        corpus = "The FOX jumps. Fox is quick.\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["fox"]
        vocab_set = set(re.findall(r"\b[a-z]+\b", corpus.lower()))

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=1)

        # tokens: the fox jumps fox is quick
        # fox(1): {the, jumps}; fox(3): {jumps, is}
        expected = Counter({"the": 1, "jumps": 2, "is": 1})
        self.assertEqual(ctx["fox"], expected)

    def test_context_counts(self):
        """Test that context word counts are positive integers."""
        corpus = "a b c a b c a\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["a"]
        vocab_set = {"a", "b", "c"}

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=2)

        for w, cnt in ctx["a"].items():
            self.assertIsInstance(cnt, int)
            self.assertGreater(cnt, 0)

    def test_context_types(self):
        """Test that contexts are Counter objects."""
        corpus = "one two three\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["one", "two"]
        vocab_set = {"one", "two", "three"}

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=1)

        self.assertIsInstance(ctx, dict)
        for t in targets:
            self.assertIn(t, ctx)
            self.assertIsInstance(ctx[t], Counter)

    def test_empty_vocab_set(self):
        """Test behavior with empty vocabulary set."""
        corpus = "the quick brown fox\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["fox"]
        vocab_set = set()

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=2)

        self.assertEqual(ctx["fox"], Counter())

    def test_multiple_target_words(self):
        """Test extraction for multiple target words."""
        corpus = "cats chase mice and dogs chase cats\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["cats", "dogs"]
        vocab_set = {"cats", "dogs", "chase", "mice", "and"}

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=1)

        self.assertEqual(ctx["cats"], Counter({"chase": 2}))
        self.assertEqual(ctx["dogs"], Counter({"and": 1, "chase": 1}))

    def test_nonexistent_word(self):
        """Test behavior with word not in corpus."""
        corpus = "the quick brown fox\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["unicorn"]
        vocab_set = {"the", "quick", "brown", "fox", "unicorn"}

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=2)

        self.assertEqual(ctx["unicorn"], Counter())

    def test_self_exclusion(self):
        """Test that target word doesn't appear in its own contexts."""
        corpus = "fox fox fox\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["fox"]
        vocab_set = {"fox"}

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=1)

        self.assertNotIn("fox", ctx["fox"])

    def test_special_characters_ignored(self):
        """Test that special characters are ignored."""
        corpus = "king!!! queen?? king's crown; queen-king.\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["king"]
        vocab_set = {"queen", "crown"}  # deliberately exclude "s"

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=1)

        # Ensure we only count clean alpha tokens that are in vocab_set
        self.assertTrue(all(re.fullmatch(r"[a-z]+", w) for w in ctx["king"].keys()))
        self.assertNotIn("s", ctx["king"])
        self.assertTrue(set(ctx["king"].keys()).issubset(vocab_set))

    def test_vocab_filtering(self):
        """Test that only words in vocab_set appear in contexts."""
        corpus = "the quick brown fox jumps over the lazy dog\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["fox"]
        vocab_set = {"quick", "jumps"}  # filter out brown/over/etc.

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=2)

        self.assertEqual(ctx["fox"], Counter({"quick": 1, "jumps": 1}))
        self.assertTrue(set(ctx["fox"].keys()).issubset(vocab_set))

    def test_window_size_effect(self):
        """Test that window size affects context extraction."""
        corpus = "the quick brown fox jumps over the lazy dog\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["fox"]
        vocab_set = set(re.findall(r"\b[a-z]+\b", corpus.lower()))

        with contextlib.redirect_stdout(io.StringIO()):
            ctx_w1 = extract_word_contexts(path, targets, vocab_set, window=1)
            ctx_w2 = extract_word_contexts(path, targets, vocab_set, window=2)

        # window=1 should not include 'quick' and 'over' for 'fox'
        self.assertNotIn("quick", ctx_w1["fox"])
        self.assertNotIn("over", ctx_w1["fox"])
        # window=2 should include them
        self.assertIn("quick", ctx_w2["fox"])
        self.assertIn("over", ctx_w2["fox"])
        self.assertGreaterEqual(len(ctx_w2["fox"]), len(ctx_w1["fox"]))

    def test_window_zero(self):
        """Test behavior with window=0 (should extract no contexts)."""
        corpus = "a b a b a\n"
        path = self._write_tmp_corpus(corpus)

        targets = ["a"]
        vocab_set = {"a", "b"}

        with contextlib.redirect_stdout(io.StringIO()):
            ctx = extract_word_contexts(path, targets, vocab_set, window=0)

        self.assertEqual(ctx["a"], Counter())


class TestComputeFitness(unittest.TestCase):
    """Unit tests for compute_fitness()."""

    def setUp(self):
        self.stats = {"mean_norm": 1.0, "std_norm": 0.25, "global_std": 1.0}
        self.dim = 4

        # Simple, stable vectors
        self.vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ctx_vecs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.neg_vecs = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.anchor_vecs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def test_context_weights_normalization(self):
        """Test that context weights don't need to be pre-normalized."""
        w_counts = np.array([2.0, 1.0], dtype=np.float32)
        w_norm = w_counts / w_counts.sum()

        weights = {"corpus": 1.0, "norm": 0.0, "anchor": 0.0}

        # Use no negatives to isolate positive weighting behavior
        neg_empty = np.empty((0, self.dim), dtype=np.float32)

        f_counts = compute_fitness(
            self.vec, "x", self.ctx_vecs, w_counts, neg_empty, None, self.stats, weights
        )
        f_norm = compute_fitness(
            self.vec, "x", self.ctx_vecs, w_norm, neg_empty, None, self.stats, weights
        )

        self.assertTrue(np.isfinite(f_counts))
        self.assertTrue(np.isfinite(f_norm))
        self.assertAlmostEqual(f_counts, f_norm, places=6)

    def test_fitness_no_nan_or_inf(self):
        """Test that fitness never produces NaN or Inf."""
        weights = {"corpus": 0.5, "norm": 0.3, "anchor": 0.2}
        ctx_w = np.array([0.5, 0.5], dtype=np.float32)

        f = compute_fitness(
            self.vec, "x", self.ctx_vecs, ctx_w, self.neg_vecs, self.anchor_vecs, self.stats, weights
        )
        self.assertTrue(np.isfinite(f))

    def test_fitness_range(self):
        """Test that fitness is in valid range [0, 1]."""
        weights = {"corpus": 0.5, "norm": 0.3, "anchor": 0.2}
        ctx_w = np.array([0.5, 0.5], dtype=np.float32)

        f = compute_fitness(
            self.vec, "x", self.ctx_vecs, ctx_w, self.neg_vecs, self.anchor_vecs, self.stats, weights
        )
        self.assertGreaterEqual(f, 0.0)
        self.assertLessEqual(f, 1.0)

    def test_fitness_with_no_anchors(self):
        """Test fitness computation when no anchor vectors provided."""
        weights = {"corpus": 0.6, "norm": 0.4, "anchor": 0.0}
        ctx_w = np.array([0.5, 0.5], dtype=np.float32)

        f = compute_fitness(
            self.vec, "x", self.ctx_vecs, ctx_w, self.neg_vecs, None, self.stats, weights
        )
        self.assertTrue(np.isfinite(f))
        self.assertTrue(0.0 <= f <= 1.0)

    def test_fitness_with_no_contexts(self):
        """Test fitness computation when no context vectors provided."""
        weights = {"corpus": 0.6, "norm": 0.2, "anchor": 0.2}

        f = compute_fitness(
            self.vec, "x", None, None, self.neg_vecs, self.anchor_vecs, self.stats, weights
        )
        self.assertTrue(np.isfinite(f))
        self.assertTrue(0.0 <= f <= 1.0)

    def test_norm_match_term(self):
        """Test that vectors with norm close to mean_norm get higher fitness."""
        weights = {"corpus": 0.0, "norm": 1.0, "anchor": 0.0}

        close = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # norm=1
        far = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)    # norm=3

        f_close = compute_fitness(close, "x", None, None, self.neg_vecs, None, self.stats, weights)
        f_far = compute_fitness(far, "x", None, None, self.neg_vecs, None, self.stats, weights)

        self.assertGreater(f_close, f_far)

    def test_weight_combination(self):
        """Test that different weight combinations produce different fitness."""
        ctx_w = np.array([0.5, 0.5], dtype=np.float32)

        w1 = {"corpus": 1.0, "norm": 0.0, "anchor": 0.0}
        w2 = {"corpus": 0.0, "norm": 0.5, "anchor": 0.5}

        f1 = compute_fitness(self.vec, "x", self.ctx_vecs, ctx_w, self.neg_vecs, self.anchor_vecs, self.stats, w1)
        f2 = compute_fitness(self.vec, "x", self.ctx_vecs, ctx_w, self.neg_vecs, self.anchor_vecs, self.stats, w2)

        self.assertNotAlmostEqual(f1, f2, places=6)

    def test_zero_norm_vector(self):
        """Test handling of zero-norm vector."""
        z = np.zeros(self.dim, dtype=np.float32)
        weights = {"corpus": 0.5, "norm": 0.3, "anchor": 0.2}

        f = compute_fitness(z, "x", self.ctx_vecs, np.array([1.0, 1.0]), self.neg_vecs, self.anchor_vecs, self.stats, weights)
        self.assertTrue(np.isfinite(f))
        self.assertTrue(0.0 <= f <= 1.0)


class TestInitializeEmbedding(unittest.TestCase):
    """Unit tests for initialize_embedding()."""

    def setUp(self):
        np.random.seed(0)
        self.vocab = [f"w{i}" for i in range(50)]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.embeddings = np.random.randn(len(self.vocab), 8).astype(np.float32)

    def test_all_contexts_missing(self):
        """Test when all context words are missing from vocabulary."""
        contexts = {"new": Counter({"missing1": 5, "missing2": 3})}
        v = initialize_embedding("new", contexts, self.embeddings, self.word_to_idx)
        self.assertTrue(np.allclose(v, self.embeddings.mean(axis=0), atol=1e-6))

    def test_different_words_different_init(self):
        """Test that different words get different initializations."""
        contexts = {
            "a": Counter({"w1": 10, "w2": 1}),
            "b": Counter({"w3": 10, "w4": 1}),
        }
        va = initialize_embedding("a", contexts, self.embeddings, self.word_to_idx)
        vb = initialize_embedding("b", contexts, self.embeddings, self.word_to_idx)
        self.assertFalse(np.allclose(va, vb))

    def test_empty_context_fallback(self):
        """Test fallback to mean embedding when no contexts."""
        contexts = {"new": Counter()}
        v = initialize_embedding("new", contexts, self.embeddings, self.word_to_idx)
        self.assertTrue(np.allclose(v, self.embeddings.mean(axis=0), atol=1e-6))

    def test_initialization_shape(self):
        """Test that initialized embedding has correct shape."""
        contexts = {"new": Counter({"w1": 1, "w2": 1})}
        v = initialize_embedding("new", contexts, self.embeddings, self.word_to_idx)
        self.assertEqual(v.shape, (self.embeddings.shape[1],))

    def test_initialization_type(self):
        """Test that initialized embedding is numpy array."""
        contexts = {"new": Counter({"w1": 1})}
        v = initialize_embedding("new", contexts, self.embeddings, self.word_to_idx)
        self.assertIsInstance(v, np.ndarray)

    def test_missing_context_words(self):
        """Test handling when context words are not in word_to_idx."""
        contexts = {"new": Counter({"w1": 2, "missing": 100})}
        v = initialize_embedding("new", contexts, self.embeddings, self.word_to_idx)
        self.assertTrue(np.isfinite(v).all())
        self.assertFalse(np.allclose(v, self.embeddings.mean(axis=0)))  # should use w1

    def test_no_nan_or_inf(self):
        """Test that initialization doesn't produce NaN or Inf."""
        contexts = {"new": Counter({"w1": 2, "w2": 3})}
        v = initialize_embedding("new", contexts, self.embeddings, self.word_to_idx)
        self.assertTrue(np.isfinite(v).all())

    def test_nonzero_norm(self):
        """Test that initialized vector has non-zero norm."""
        contexts = {"new": Counter({"w1": 2, "w2": 3})}
        v = initialize_embedding("new", contexts, self.embeddings, self.word_to_idx)
        self.assertGreater(float(np.linalg.norm(v)), 0.0)

    def test_top_k_contexts_used(self):
        """Test that only top contexts are used (max 20)."""
        # Make embeddings easy to reason about: each word has a unique one-hot-ish vector
        dim = 8
        emb = np.zeros((50, dim), dtype=np.float32)
        for i in range(50):
            emb[i, i % dim] = float(i + 1)  # unique contribution per word

        word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        # 30 contexts: top 20 are w0..w19 with high counts, bottom 10 are w20..w29 with low counts
        ctx = Counter()
        for i in range(20):
            ctx[f"w{i}"] = 100 - i
        for i in range(20, 30):
            ctx[f"w{i}"] = 1

        contexts = {"new": ctx}
        v = initialize_embedding("new", contexts, emb, word_to_idx)

        # Compute expected using only top 20
        top20 = ctx.most_common(20)
        denom = sum(c for _, c in top20)
        expected = np.zeros(dim, dtype=np.float32)
        for w, c in top20:
            expected += emb[word_to_idx[w]] * (c / denom)

        self.assertTrue(np.allclose(v, expected, atol=1e-6))

    def test_weighted_average(self):
        """Test that initialization uses weighted average of contexts."""
        emb = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        word_to_idx = {"a": 0, "b": 1, "c": 2}
        contexts = {"x": Counter({"a": 2, "b": 1, "c": 1})}
        v = initialize_embedding("x", contexts, emb, word_to_idx)
        expected = np.array([0.5, 0.25, 0.25], dtype=np.float32)
        self.assertTrue(np.allclose(v, expected, atol=1e-6))


class TestPrecomputeFitnessVectors(unittest.TestCase):
    """Unit tests for precompute_fitness_vectors()."""

    def setUp(self):
        np.random.seed(1)
        self.vocab = [f"w{i}" for i in range(80)]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.embeddings = np.random.randn(len(self.vocab), 10).astype(np.float32)

    def test_anchor_vectors_normalized(self):
        """Test that anchor vectors are normalized."""
        contexts = {"x": Counter({"w1": 2})}
        anchors = {"x": ["w2", "w3", "w4"]}
        ctx_v, ctx_w, neg_v, anc_v = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=7
        )
        self.assertIsNotNone(anc_v)
        norms = np.linalg.norm(anc_v, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_context_vectors_shape(self):
        """Test that context vectors have correct shape."""
        contexts = {"x": Counter({"w1": 3, "w2": 1, "missing": 9})}
        anchors = {"x": []}
        ctx_v, ctx_w, neg_v, anc_v = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsNotNone(ctx_v)
        self.assertEqual(ctx_v.shape[1], self.embeddings.shape[1])
        self.assertEqual(ctx_v.shape[0], 2)  # w1, w2 only

    def test_context_weights_match_vectors(self):
        """Test that number of weights matches number of vectors."""
        contexts = {"x": Counter({"w1": 3, "w2": 1})}
        anchors = {}
        ctx_v, ctx_w, _, _ = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsNotNone(ctx_v)
        self.assertIsNotNone(ctx_w)
        self.assertEqual(ctx_v.shape[0], ctx_w.shape[0])

    def test_context_weights_positive(self):
        """Test that all context weights are positive."""
        contexts = {"x": Counter({"w1": 3, "w2": 1})}
        anchors = {}
        _, ctx_w, _, _ = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsNotNone(ctx_w)
        self.assertTrue(np.all(ctx_w > 0))

    def test_context_weights_sum(self):
        """Test that context weights are normalized."""
        contexts = {"x": Counter({"w1": 3, "w2": 1})}
        anchors = {}
        _, ctx_w, _, _ = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsNotNone(ctx_w)
        self.assertAlmostEqual(float(ctx_w.sum()), 1.0, places=6)

    def test_empty_anchors(self):
        """Test handling of empty anchors."""
        contexts = {"x": Counter({"w1": 3})}
        anchors = {"x": []}
        _, _, _, anc_v = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsNone(anc_v)

    def test_empty_contexts(self):
        """Test handling of empty contexts."""
        contexts = {"x": Counter()}
        anchors = {}
        ctx_v, ctx_w, _, _ = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsNone(ctx_v)
        self.assertIsNone(ctx_w)

    def test_missing_anchor_words(self):
        """Test handling when anchor words not in vocabulary."""
        contexts = {"x": Counter({"w1": 1})}
        anchors = {"x": ["missing1", "missing2"]}
        _, _, _, anc_v = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsNone(anc_v)

    def test_missing_context_words(self):
        """Test handling when context words not in vocabulary."""
        contexts = {"x": Counter({"missing": 10})}
        anchors = {}
        ctx_v, ctx_w, _, _ = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsNone(ctx_v)
        self.assertIsNone(ctx_w)

    def test_negative_samples_randomness(self):
        """Test that negative samples are random across calls."""
        contexts = {"x": Counter({"w1": 1})}
        anchors = {}
        np.random.seed(123)  # deterministic, but consecutive calls should differ

        _, _, neg1, _ = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=10
        )
        _, _, neg2, _ = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=10
        )

        self.assertFalse(np.array_equal(neg1, neg2))

    def test_negative_vectors_count(self):
        """Test that correct number of negative samples returned."""
        contexts = {"x": Counter({"w1": 1})}
        anchors = {}
        _, _, neg_v, _ = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=13
        )
        self.assertEqual(neg_v.shape, (13, self.embeddings.shape[1]))

    def test_no_nan_in_outputs(self):
        """Test that no NaN values in any output."""
        contexts = {"x": Counter({"w1": 3, "w2": 1})}
        anchors = {"x": ["w3", "w4"]}
        ctx_v, ctx_w, neg_v, anc_v = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=9
        )

        for arr in (ctx_v, ctx_w, neg_v, anc_v):
            if arr is None:
                continue
            self.assertFalse(np.isnan(arr).any())

    def test_return_tuple_length(self):
        """Test that function returns tuple of 4 elements."""
        contexts = {"x": Counter({"w1": 1})}
        anchors = {}
        out = precompute_fitness_vectors(
            "x", contexts, self.embeddings, self.word_to_idx, self.vocab, anchors, num_negatives=5
        )
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)


def run_tests():
    """Run ONLY the Lab 7 unit tests requested and return success as bool."""
    print("\n" + "=" * 70)
    print("RUNNING LAB 7 UNIT TESTS")
    print("=" * 70)

    suite = unittest.TestSuite()
    for cls in (
        TestExtractWordContexts,
        TestComputeFitness,
        TestInitializeEmbedding,
        TestPrecomputeFitnessVectors,
    ):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()



if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)    
