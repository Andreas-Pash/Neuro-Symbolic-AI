import argparse
from collections import Counter
from urllib.parse import urlparse
import requests
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def load_from_source(source):
    """
    Load data from file path or URL.
    
    Args:
        source: File path (str) or URL (str)
    
    Returns:
        bytes: Raw content (text or binary)
    
    Raises:
        requests.HTTPError: If URL request fails
        FileNotFoundError: If file does not exist
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        response = requests.get(source)
        response.raise_for_status()
        return response.content
    else:
        with open(source, "rb") as f:
            return f.read()


def build_unweighted_graph(nodes, adjacency_counts):
    """
    Build an unweighted NetworkX graph from adjacency counts.

    This function creates a graph where nodes are connected by edges if they
    appear adjacent in the text. The edge weights (how many times tokens appear
    together) are ignored - we only care about whether two tokens ever appear
    next to each other, not how often.

    Think of it like a social network: if two people have ever talked, there's
    a connection between them, regardless of how many conversations they've had.

    Important: This creates an UNDIRECTED graph, meaning if A is connected to B,
    then B is automatically connected to A. We also avoid duplicate edges - each
    unique pair gets exactly one edge.

    Parameters:
    -----------
    nodes : list
        List of node identifiers (e.g., tokens/words to include in the graph)
    adjacency_counts : Counter
        Mapping of (node1, node2) tuples to their frequency counts
        Example: {('cat', 'dog'): 3, ('dog', 'bird'): 1}

    Returns:
    --------
    networkx.Graph
        An undirected, unweighted graph with edges between adjacent nodes

    Examples:
    ---------
    >>> nodes = ['cat', 'dog', 'bird']
    >>> adjacency_counts = Counter({('cat', 'dog'): 3, ('dog', 'bird'): 1})
    >>> G = build_unweighted_graph(nodes, adjacency_counts)
    >>> list(G.edges())
    [('cat', 'dog'), ('dog', 'bird')]
    >>> G.number_of_nodes()
    3

    Algorithm Steps:
    ----------------
    1. Create empty NetworkX Graph
    2. Add all nodes to the graph
    3. Loop through adjacency_counts to find pairs
    4. For each pair, check if both nodes are valid and edge doesn't exist
    5. Add edge between valid node pairs
    6. Return the completed graph
    """ 

    # STEP 1: Create an empty NetworkX Graph (undirected)
    G = nx.Graph() 

    # STEP 2: Add all nodes to the graph
    G.add_nodes_from(nodes)
    
    # STEP 3: Loop through adjacency_counts to process each pair
    for (a, b), count in adjacency_counts.items():
        if (a in nodes) and (b in nodes) and (not G.has_edge(a, b)):  
            G.add_edge(a, b)
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
 
    return G


def compute_distance_matrix(nodes, adjacency_counts, distance_mode="inverted"):
    """
    Compute a distance matrix from adjacency counts between nodes.

    This function takes counts of how often nodes (tokens) appear next to each other
    and converts them into a distance matrix suitable for network analysis algorithms.

    The key insight: tokens that appear together frequently should be "close" in
    distance, while tokens that never appear together should be "far apart."

    Important Transformations:
    --------------------------
    1. SYMMETRIZATION: Adjacency counts may be directional (A->B might differ from B->A),
       but we create an undirected representation by averaging: (count(A,B) + count(B,A)) / 2

    2. SELF-CONNECTIONS: The diagonal (distance from a node to itself) is always zero

    3. DISTANCE MODES:
       - 'direct': Higher count = Higher distance (rarely used)
       - 'inverted': Higher count = Lower distance (default, more intuitive)
         Formula: distance = (max_count + 1) - count

    Think of it like a city map: if two neighborhoods share many connections (high count),
    they're close together (low distance). If they never connect, they're far apart.

    Parameters:
    -----------
    nodes : list
        Ordered list of node identifiers (defines matrix row/column order)
        Example: ['cat', 'dog', 'bird'] -> cat is index 0, dog is 1, bird is 2
    
    adjacency_counts : Counter
        Mapping of (node1, node2) -> frequency (may be directional)
        Example: {('cat', 'dog'): 5, ('dog', 'cat'): 3, ('dog', 'bird'): 2}
    
    distance_mode : str, default='inverted'
        How to convert counts to distances:
        - 'direct': distance = count (higher count = farther apart)
        - 'inverted': distance = (max_count + 1) - count (higher count = closer)

    Returns:
    --------
    tuple of (distance_matrix, count_matrix)
        distance_matrix : numpy array (n x n)
            Symmetric distance matrix with zero diagonal
        count_matrix : numpy array (n x n)
            Raw symmetrized adjacency counts (before distance conversion)

    Raises:
    -------
    ValueError
        If distance_mode is not 'direct' or 'inverted'

    Examples:
    ---------
    >>> nodes = ['A', 'B', 'C']
    >>> adjacency_counts = Counter({('A', 'B'): 4, ('B', 'A'): 2, ('B', 'C'): 1})
    >>> dist_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, 'inverted')
    >>> print(count_matrix)
    [[0.  3.  0. ]   # A-B averaged: (4+2)/2 = 3
     [3.  0.  1. ]   # B-C: 1 (no reverse)
     [0.  1.  0. ]]  # Symmetric
    >>> print(dist_matrix)  # max_count=3, so (3+1)-count = 4-count
    [[0.  1.  4. ]   # A-B: 4-3=1 (close), A-C: 4-0=4 (far)
     [1.  0.  3. ]
     [4.  3.  0. ]]

    Algorithm Steps:
    ----------------
    1. Get number of nodes and create node-to-index mapping
    2. Create empty count matrix (n x n)
    3. Fill matrix with directional counts from adjacency_counts
    4. Symmetrize the matrix by averaging with its transpose
    5. Remove self-connections (set diagonal to zero)
    6. Convert counts to distances based on distance_mode
    7. Ensure diagonal is zero in distance matrix
    8. Return both distance_matrix and count_matrix
    """

    n = len(nodes)
    node_index = {node:i for i,node in enumerate(nodes)}  

    print(f"Processing {n} nodes: {nodes}")
    print(f"Node index mapping: {node_index}")

    count_matrix = np.zeros((n,n))
    for (a, b), count in adjacency_counts.items():
        print((a,b))
        if (a in nodes) and (b in nodes):
            count_matrix[node_index[a],node_index[b]] = count 

    print(f"Filled count matrix with directional counts")
    print(f"Count matrix (before symmetrization):\n{count_matrix}")

    # Symmetrize the matrix for undirected representation
    count_matrix = (count_matrix + count_matrix.T) / 2
    print(f"Symmetrized count matrix:\n{count_matrix}")
    
    np.fill_diagonal(count_matrix, 0)
    print(f"Removed self-connections (diagonal set to 0)")

    # STEP 6: Convert counts to distances based on distance_mode

    if distance_mode == "direct":
        distance_matrix = count_matrix.copy()   
    elif distance_mode == "inverted":   
        # higher count -> lower distance
        max_count = np.max(count_matrix)
        print(f"Max count in matrix: {max_count}")

        if max_count > 0:
            distance_matrix = (max_count + 1) - count_matrix   
            print(f"Using inverted distance: (max_count + 1) - count")
        else:
            # No edges exist: set all distances to 1 (except diagonal which is 0)
            distance_matrix = np.ones_like(count_matrix)
            print(f"No edges found, setting all distances to 1")
    else:
        raise ValueError("distance_mode must be 'direct' or 'inverted'")
        
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix, count_matrix


def visualize_network(G, distance_matrix, nodes, node_colors=None, node_labels=None,
                       figsize=(14, 14), title="Network Graph", jitter=0.2, random_state=42):
    """
    Visualize network in matrix-style grid layout with edge distance labels.
    
    Nodes are arranged in a grid pattern with small random jitter to prevent
    edge overlap. Edge labels show distance values from the distance matrix.
    
    Args:
        G: NetworkX graph
        distance_matrix: numpy array of distances (n x n)
        nodes: ordered list of nodes (must match distance_matrix order)
        node_colors: list of colors for nodes (default: all 'lightblue')
        node_labels: dict of node -> label string (default: str(node)[:10])
        figsize: matplotlib figure size tuple
        title: plot title string
        jitter: max random offset for node positions (default: 0.2)
        random_state: random seed for reproducible jitter (default: 42)
    """
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    side = int(np.ceil(np.sqrt(n)))
    
    # Grid positions with jitter to avoid edge overlap
    pos = {}
    rng = np.random.RandomState(random_state)
    for i, node in enumerate(nodes):
        row = i // side
        col = i % side
        jitter_x = rng.uniform(-jitter, jitter)
        jitter_y = rng.uniform(-jitter, jitter)
        pos[node] = (col + jitter_x, -row + jitter_y)
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(title)
    
    # Default node properties
    if node_colors is None:
        node_colors = ["lightblue"] * n
    if node_labels is None:
        node_labels = {node: str(node)[:10] for node in nodes}
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors,
                           edgecolors="black", linewidths=1, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7, ax=ax)
    
    # Edge labels (distances)
    edge_labels = {}
    for u, v in G.edges():
        if u in node_index and v in node_index:
            dist_val = distance_matrix[node_index[u], node_index[v]]
            edge_labels[(u, v)] = f"{dist_val:.1f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax, label_pos=0.5)
    
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# TEXT PROCESSING
# =============================================================================


def tokenize_text(text, char_like=["'"]):
    """
    Tokenize text into individual words and punctuation marks.

    This function breaks down a text string into meaningful pieces (tokens).
    It separates words from punctuation and handles various text characters.
    Think of it like breaking a sentence into individual puzzle pieces.

    Rules:
    ------
    - Words: sequences of alphabetic characters (a-z, A-Z, Unicode letters)
    - Punctuation: comma (,) and period (.) are kept as separate tokens
    - Characters in `char_like` (e.g., apostrophes) are treated as part of words
    - Other characters: spaces, numbers, symbols act as word separators (ignored)
    - Case: everything is converted to lowercase

    Parameters:
    -----------
    text : str
        Input text to tokenize (e.g., "Hello, world! How are you?")
    char_like : list
        Optional characters that should be treated as if they were alphabet letters

    Returns:
    --------
    list of str
        List of tokens (words and punctuation marks)

    Examples:
    ---------
    >>> tokenize_text("Hello, world!")
    ['hello', ',', 'world']

    >>> tokenize_text("I like cats, dogs, and birds.")
    ['i', 'like', 'cats', ',', 'dogs', ',', 'and', 'birds', '.']

    >>> tokenize_text("Numbers123 and symbols@#$ are ignored")
    ['numbers', 'and', 'symbols', 'are', 'ignored']

    Algorithm Overview:
    ------------------
    1. Convert text to lowercase
    2. Scan character by character
    3. Build words by collecting alphabetic characters
    4. When we hit punctuation or separators, finish the current word
    5. Keep commas and periods as tokens, ignore other separators
    """
 
    # STEP 1: Prepare the text and initialize variables
    text = text.lower()
    tokens = []
    current_word = []
    punctuation = {',', '.'} 


    def flush_word():
        nonlocal current_word
        if current_word:
            tokens.append(''.join(current_word))
            current_word = []
    for char in text:
        if char.isalpha() or char in char_like:
            current_word.append(char)
        elif char in punctuation:
            flush_word()
            tokens.append(char)
        else:
            flush_word()

    if current_word:
        tokens.append(''.join(current_word))

    return tokens


def replace_rare_tokens(tokens, rare_threshold=0.01, rare_token="<RARE>", must_have_tokens = []):
    """
    Replace infrequent words with a special rare token to reduce vocabulary size.

    This function helps clean up text data by replacing words that appear very rarely
    with a single special token. This is useful in text analysis because rare words
    often don't provide much meaningful information but can make datasets harder to work with.

    Important: Only words are considered for replacement, punctuation (like , and .)
    is always kept as-is regardless of frequency.

    Parameters:
    -----------
    tokens : list of str
        List of tokens from tokenized text (e.g., ['hello', ',', 'world', '.'])
    rare_threshold : float, default=0.01
        Fractional threshold between 0 and 1.
        Words appearing less than this fraction of total words will be replaced.
    rare_token : str, default="<RARE>"
        The special token to use as replacement for rare words.

    Returns:
    --------
    tuple of (new_tokens, rare_set, final_counts)
        new_tokens : list of str
            Original token list with rare words replaced
        rare_set : set of str
            Set of original words that were considered rare and replaced
        final_counts : Counter
            Frequency count of tokens in the final new_tokens list

    Example:
    --------
    >>> tokens = ['cat', 'dog', 'cat', 'elephant', 'dog', 'cat', '.']
    >>> new_tokens, rare_set, final_counts = replace_rare_tokens(tokens, rare_threshold=0.25)
    >>> print(f"Rare words: {rare_set}")
    {'elephant'}

    Algorithm Steps:
    ----------------
    1. Filter out punctuation to count only actual words
    2. Calculate what proportion each word represents
    3. Identify words below the threshold as "rare"
    4. Replace rare words in original token list
    5. Count final token frequencies
    """
    
    punctuation = {',', '.'}
    word_tokens = [t for t in tokens if t not in punctuation]
    total_words = len(word_tokens)

    if total_words == 0:
        return tokens[:], set(), Counter(tokens)

    word_counts = Counter(word_tokens)
    rare_set = {word for word, count in word_counts.items()
            if (count / total_words < rare_threshold) and (word not in must_have_tokens) }

    new_tokens = [
        rare_token if (t not in punctuation and t in rare_set) else t
        for t in tokens
    ]
    final_counts = Counter(new_tokens)
    
    return new_tokens, rare_set, final_counts


def get_text_adjacencies(tokens):
    """
    Count how often each pair of consecutive tokens appears together in the text.

    This function creates a "bigram" analysis - it looks at every pair of adjacent
    tokens and counts how many times each pair occurs. This helps us understand
    which words tend to follow other words in the text.

    Think of it like analyzing conversation patterns: if someone says "good",
    how often is the next word "morning" vs "afternoon" vs "luck"?

    Parameters:
    -----------
    tokens : list of str
        List of tokens in order (e.g., ['the', 'cat', 'sat', 'on', 'the', 'mat'])

    Returns:
    --------
    Counter
        Dictionary-like object mapping (token1, token2) tuples to their frequency count
        Key: (current_token, next_token)
        Value: number of times this pair appears consecutively

    Examples:
    ---------
    >>> tokens = ['the', 'cat', 'sat', 'on', 'the', 'cat']
    >>> adjacencies = get_text_adjacencies(tokens)
    >>> print(dict(adjacencies))
    {('the', 'cat'): 2, ('cat', 'sat'): 1, ('sat', 'on'): 1, ('on', 'the'): 1}

    Explanation:
    - 'the' is followed by 'cat' twice: positions (0,1) and (4,5)
    - 'cat' is followed by 'sat' once: position (1,2)
    - 'sat' is followed by 'on' once: position (2,3)
    - 'on' is followed by 'the' once: position (3,4)

    Visual Example:
    ---------------
    tokens = ['I', 'love', 'cats', '.', 'I', 'love', 'dogs']
    pairs:    ^      ^      ^     ^    ^      ^
             (0,1)  (1,2)  (2,3) (3,4) (4,5) (5,6)

    Results: ('I', 'love'): 2, ('love', 'cats'): 1, ('cats', '.'): 1,
             ('.', 'I'): 1, ('love', 'dogs'): 1

    Algorithm Steps:
    ----------------
    1. Initialize Counter to store adjacency counts
    2. Loop through token positions 0 to len(tokens)-2
    3. For each position i, get tokens[i] and tokens[i+1]
    4. Only count pairs where the tokens are different (no self-loops)
    5. Create tuple pair (current_token, next_token) and increment count
    6. Return the Counter with all pair frequencies
    """

    # STEP 1: Initialize Counter to store adjacency counts
    adjacency_counts = Counter()
    
    for i in range(len(tokens) - 1):
        current_token = tokens[i]
        next_token = tokens[i + 1]
        if current_token != next_token:
            adjacency_counts[(current_token, next_token)] += 1

    return adjacency_counts


def process_text_network(source, rare_threshold=0.01, rare_token="<RARE>", must_have_tokens = [],
                          distance_mode="inverted", verbose=True, nsample_tokens=20):
    """
    Complete text network processing pipeline.
    
    Pipeline:
    1. Load text from source (file or URL)
    2. Tokenize into words and punctuation
    3. Replace rare tokens
    4. Build directional adjacency counts from consecutive tokens
    5. Create unweighted graph and distance matrix
    
    Args:
        source: file path or URL
        rare_threshold: token rarity threshold (0-1)
        rare_token: replacement string for rare tokens
        distance_mode: 'direct' or 'inverted' (see compute_distance_matrix)
        verbose: if True, print processing details
        nsample_tokens: number of sample tokens to print (if verbose)
    
    Returns:
        dict with keys:
            - graph: NetworkX Graph
            - nodes: list of node identifiers (sorted by frequency)
            - adjacency_counts: Counter of (node1, node2) -> count
            - distance_matrix: numpy array (n x n)
            - count_matrix: symmetrized adjacency counts
            - token_counts: Counter of final token frequencies
            - rare_tokens: set of replaced tokens
            - original_tokens: list of tokens before rare replacement
    """
    # Load and tokenize
    content = load_from_source(source)
    text = content.decode('utf-8', errors='ignore')
    
    if verbose:
        print(f"Loaded text: {len(text)} characters")
    
    tokens = tokenize_text(text, char_like= ["'", "_"])
    
    if verbose:
        print(f"Tokenized: {len(tokens)} tokens")
        print(f"Sample tokens: {list(set(tokens))[:nsample_tokens]}")
    
    # Handle rare tokens
    processed_tokens, rare_set, token_counts = replace_rare_tokens(
        tokens, rare_threshold, rare_token, must_have_tokens)
    
    if verbose:
        print(f"Replaced {len(rare_set)} rare tokens (threshold={rare_threshold})")
        print(f"Final vocabulary: {len(token_counts)} unique tokens")
        print(f"Sample tokens: {list(set(processed_tokens))[:nsample_tokens]}")
    
    # Build adjacencies and graph
    adjacency_counts = get_text_adjacencies(processed_tokens)
    nodes = sorted(token_counts.keys(), key=lambda x: (-token_counts[x], x))
    graph = build_unweighted_graph(nodes, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, distance_mode)
    
    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Top tokens by frequency:")
        for i, node in enumerate(nodes[:10]):
            print(f"  {i+1:2d}. '{node}' (freq={token_counts[node]})")
    
    return {
        'graph': graph,
        'nodes': nodes,
        'adjacency_counts': adjacency_counts,
        'distance_matrix': distance_matrix,
        'count_matrix': count_matrix,
        'token_counts': token_counts,
        'rare_tokens': rare_set,
        'original_tokens': tokens
    }


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def preprocess_image(image_data, target_size=(128, 128), quantize_levels=16):
    """
    Load and preprocess image: resize and quantize colors.
    
    Quantization maps each pixel channel value to the nearest of N evenly-spaced
    levels. For uniform channels (min == max), all pixels map to level 0.
    
    Performance note: Uses vectorized numpy operations for efficient quantization.
    
    Args:
        image_data: raw image bytes
        target_size: resize dimensions (width, height)
        quantize_levels: number of quantization levels per channel
    
    Returns:
        tuple of:
            - quantized_image: numpy array of quantized indices (height x width x channels)
            - quantization_info: dict mapping channel_idx -> {'min', 'max', 'levels'}
                where 'levels' is the array of quantization boundary values
    """
    img = Image.open(BytesIO(image_data))
    
    # Handle different image modes
    if img.mode == 'RGBA':
        # Composite with white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Add channel dimension for grayscale
    if len(img_array.shape) == 2:
        img_array = img_array[:, :, np.newaxis]
    
    height, width, channels = img_array.shape
    quantized = np.zeros_like(img_array)
    quantization_info = {}
    
    # Quantize each channel independently using vectorized operations
    for c in range(channels):
        channel_data = img_array[:, :, c]
        min_val, max_val = channel_data.min(), channel_data.max()
        
        if min_val == max_val:
            # Uniform channel: all pixels map to level 0
            quantized[:, :, c] = 0
            quantization_info[c] = {'min': min_val, 'max': max_val, 'levels': [min_val]}
        else:
            # Create evenly-spaced quantization levels
            levels = np.linspace(min_val, max_val, quantize_levels)
            
            # Vectorized quantization: find nearest level index for each pixel
            # Using np.searchsorted for O(N*log(M)) instead of O(N*M)
            quantized[:, :, c] = np.searchsorted(levels, channel_data, side='left')
            # Clamp to valid range [0, quantize_levels-1]
            quantized[:, :, c] = np.clip(quantized[:, :, c], 0, quantize_levels - 1)
            
            quantization_info[c] = {'min': min_val, 'max': max_val, 'levels': levels}
    
    return quantized, quantization_info


def get_spatial_adjacencies(quantized_image):
    """
    Analyze spatial relationships between colors by examining neighboring pixels.

    This function treats an image like a graph where each pixel is a node and edges
    connect neighboring pixels. It counts how often different colors appear next to
    each other, which helps us understand the image's structure and color patterns.

    Think of it like analyzing a neighborhood: which house colors tend to be next to
    each other? Do blue houses neighbor red houses often? This is the same idea but
    for pixels.

    Important: We only count adjacencies between DIFFERENT colors. If two adjacent
    pixels have the same color, we don't record that adjacency (no self-loops).

    Parameters:
    -----------
    quantized_image : numpy.ndarray
        3D array with shape (height, width, channels) containing quantized color values
        Each pixel is represented as quantized integers (e.g., [2, 5, 1] for RGB)

    Returns:
    --------
    tuple of (adjacency_counts, unique_colors, color_frequencies)
        adjacency_counts : Counter
            Maps (color1, color2) tuples to frequency of spatial adjacency
        unique_colors : list of tuples
            All unique colors sorted by frequency (most common first)
        color_frequencies : Counter
            Maps color tuples to total pixel counts in the image

    Neighbor Definition (4-connected):
    ----------------------------------
    For each pixel at position (i, j), we check 4 neighbors:
        - Up:    (i-1, j)  ↑
        - Down:  (i+1, j)  ↓
        - Left:  (i, j-1)  ←
        - Right: (i, j+1)  →

    We ignore diagonal neighbors and boundary pixels that would go outside the image.

    Example:
    --------
    For a 2x2 image:  [[A, B],
                       [C, D]]

    Adjacencies checked (assuming all different colors):
    - Pixel A: neighbors B (right) and C (below)
    - Pixel B: neighbors A (left) and D (below)
    - Pixel C: neighbors A (above) and D (right)
    - Pixel D: neighbors B (above) and C (left)

    Color Format:
    -------------
    Colors are stored as tuples, e.g., (2, 5, 1) for a 3-channel quantized pixel.
    This allows the same analysis to work for grayscale (1 channel), RGB (3 channels),
    or other multi-channel images.

    Algorithm Steps:
    ----------------
    1. Extract image dimensions (height, width, channels)
    2. Initialize Counters for adjacencies and color frequencies
    3. Define 4-connected neighbor direction offsets
    4. Scan every pixel in the image with nested loops
    5. For each pixel, convert its color to a tuple
    6. Count the color's frequency
    7. Check all 4 neighbors using direction offsets
    8. For each neighbor, verify it's within image bounds
    9. If valid, get the neighbor's color as a tuple
    10. If colors differ, record the adjacency
    11. Sort unique colors by frequency (descending)
    12. Return all three results
    """
 
    height, width, channels = quantized_image.shape
    adjacency_counts = Counter()
    color_frequencies = Counter()

    #  Define the 4-connected neighbor directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Scan every pixel in the image with nested loops
    for i in range(height):
        for j in range(width):

            # Get current pixel's color as a tuple
            current_color = tuple(quantized_image[i, j])

            # Count this color's frequency
            color_frequencies[current_color] += 1

            # Check all 4-connected neighbors
            for di, dj in directions:

                # Calculate neighbor coordinates
                ni = i + di
                nj = j + dj

                # Check if neighbor is within image bounds
                if 0 <= ni < height and 0 <= nj < width:

                    # Get neighbor's color as a tuple
                    neighbor_color = tuple(quantized_image[ni, nj])

                    # Check if colors are different and record adjacency
                    if current_color != neighbor_color:
                        adjacency_counts[(current_color, neighbor_color)] += 1

    # Create sorted list of unique colors
    unique_colors = sorted(
        color_frequencies.keys(),
        key=lambda x: (-color_frequencies[x], x)
    )

    return adjacency_counts, unique_colors, color_frequencies


def color_to_rgb(color_tuple, quantization_info):
    """
    Convert quantized color tuple back to RGB for visualization.
    
    Maps each quantized index back to its corresponding level value,
    then normalizes to [0, 1] range for matplotlib. Handles grayscale
    by replicating the single channel to RGB.
    
    Args:
        color_tuple: tuple of quantized channel indices
        quantization_info: dict mapping channel_idx -> {'levels': array}
    
    Returns:
        RGB tuple with values in [0, 1] range
    """
    rgb = []
    for c, quant_val in enumerate(color_tuple):
        if c < len(quantization_info):
            levels = quantization_info[c]['levels']
            if isinstance(levels, (list, np.ndarray)) and len(levels) > int(quant_val):
                rgb_val = levels[int(quant_val)]
            else:
                rgb_val = levels[0] if hasattr(levels, '__getitem__') else levels
            rgb.append(rgb_val / 255.0)  # Normalize to [0,1]
        else:
            rgb.append(0.5)
    
    # Handle different channel counts
    if len(rgb) == 1:
        return (rgb[0], rgb[0], rgb[0])  # Grayscale to RGB
    elif len(rgb) >= 3:
        return tuple(rgb[:3])
    else:
        return tuple((rgb + [0, 0, 0])[:3])


def show_quantized_image(quantized_image, quantization_info, figsize=(8, 8)):
    """
    Display the quantized image.
    
    Args:
        quantized_image: numpy array of quantized indices
        quantization_info: quantization metadata from preprocess_image
        figsize: matplotlib figure size
    """
    height, width, channels = quantized_image.shape
    display_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            color_tuple = tuple(quantized_image[i, j])
            rgb = color_to_rgb(color_tuple, quantization_info)
            display_image[i, j] = [int(c * 255) for c in rgb]
    
    plt.figure(figsize=figsize)
    plt.imshow(display_image)
    plt.title(f"Quantized Image ({height}x{width}, {channels} channels)")
    plt.axis('off')
    plt.show()


def process_image_network(source, target_size=(128, 128), quantize_levels=16,
                           distance_mode="inverted", verbose=True):
    """
    Complete image network processing pipeline.
    
    Pipeline:
    1. Load image from source (file or URL)
    2. Resize and quantize colors
    3. Extract spatial adjacencies between different colors
    4. Build unweighted graph and distance matrix
    
    Args:
        source: file path or URL
        target_size: resize dimensions (width, height)
        quantize_levels: quantization levels per channel
        distance_mode: 'direct' or 'inverted' (see compute_distance_matrix)
        verbose: if True, print processing details and show quantized image
    
    Returns:
        dict with keys:
            - graph: NetworkX Graph
            - nodes: list of color tuples (sorted by frequency)
            - adjacency_counts: Counter of (color1, color2) -> count
            - distance_matrix: numpy array (n x n)
            - count_matrix: symmetrized adjacency counts
            - color_frequencies: Counter of color -> pixel count
            - quantized_image: numpy array of quantized image
            - quantization_info: quantization metadata
    """
    # Load and preprocess image
    content = load_from_source(source)
    quantized_image, quantization_info = preprocess_image(content, target_size, quantize_levels)
    
    if verbose:
        print(f"Image processed: {quantized_image.shape}, {quantize_levels} levels per channel")
    
    if verbose:
        # Show quantized image
        show_quantized_image(quantized_image, quantization_info)
    
    # Get spatial adjacencies
    adjacency_counts, unique_colors, color_frequencies = get_spatial_adjacencies(quantized_image)
    
    if verbose:
        print(f"Found {len(unique_colors)} unique colors")
        print(f"Total spatial adjacencies: {sum(adjacency_counts.values())}")
        print("Top colors by frequency:")
        for i, color in enumerate(unique_colors[:10]):
            print(f"  {i+1:2d}. {color} (freq={color_frequencies[color]})")
    
    # Build graph
    graph = build_unweighted_graph(unique_colors, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(unique_colors, adjacency_counts, distance_mode)
    
    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    return {
        'graph': graph,
        'nodes': unique_colors,
        'adjacency_counts': adjacency_counts,
        'distance_matrix': distance_matrix,
        'count_matrix': count_matrix,
        'color_frequencies': color_frequencies,
        'quantized_image': quantized_image,
        'quantization_info': quantization_info
    }


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestUnifiedNetworks(unittest.TestCase):
    """
    Unit tests for the unified_networks module.
    These are just a few examples. You should implement more. 
    You will be assessed on different tests, not the ones listed here.
    
    These tests help you verify that your implementations are working correctly.
    Each test focuses on a specific function and checks its behavior with simple inputs.
    
    Testing Tips:
    -------------
    - Run tests frequently as you implement each function
    - Read the test descriptions to understand expected behavior
    - Use print statements in your code to debug failures
    - Start with simple test cases before adding complex ones
    
    After completing these starter tests, consider adding your own tests for:
    - Edge cases (empty inputs, single elements, etc.)
    - Different parameter values (thresholds, modes, etc.)
    - Error handling (invalid inputs, out-of-bounds, etc.)
    - Integration tests (full pipeline end-to-end)
    """
    
    def test_build_unweighted_graph(self):
        """Test building an unweighted, undirected graph from adjacency counts."""
        nodes = ['A', 'B', 'C']
        adjacency_counts = Counter({('A', 'B'): 3, ('B', 'C'): 2, ('A', 'C'): 1})
        
        G = build_unweighted_graph(nodes, adjacency_counts)
                
        self.assertIsInstance(G, nx.Graph)
        
        self.assertEqual(G.number_of_nodes(), 3)    
        self.assertEqual(G.number_of_edges(), 3)
    
        self.assertTrue(G.has_edge('A', 'B'))
        self.assertTrue(G.has_edge('B', 'C'))
        self.assertTrue(G.has_edge('A', 'C'))
        
        self.assertEqual(set(G.nodes()), set(nodes))
        
    
    def test_compute_distance_matrix_inverted_mode(self):
        """
        Test distance matrix computation in inverted mode.
        
        In inverted mode, frequently co-occurring nodes should have LOW distances.
        Formula: distance = (max_count + 1) - symmetrized_count
        
        Important: The adjacency counts are symmetrized BEFORE computing distances.
        """
        
        nodes = ['A', 'B']
        adjacency_counts = Counter({('A', 'B'): 4, ('B', 'A'): 2})
        
        distance_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, mode="inverted")
        
        expected_counts = np.array([[0.0, 3.0],
                                    [3.0, 0.0]])
        np.testing.assert_array_almost_equal(count_matrix, expected_counts, decimal=6)
        

        expected_distances = np.array([[0.0, 1.0],
                                       [1.0, 0.0]])
        
        np.testing.assert_array_almost_equal(distance_matrix, expected_distances, decimal=6)
        np.testing.assert_array_equal(np.diag(count_matrix), np.array([0.0, 0.0]))
        np.testing.assert_array_equal(np.diag(distance_matrix), np.array([0.0, 0.0]))
    
    def test_tokenize_text_basic(self):
        """
        Test basic text tokenization.
        
        The tokenizer should:
        - Convert to lowercase
        - Keep letters and apostrophes as part of words
        - Keep commas and periods as separate tokens
        - Ignore numbers, symbols, and other punctuation
        """
         
        text = "Hello, world. How're you?! 123abc$%^def"
         
        tokens = tokenize_text(text) 
        expected = ["hello", ",", "world", ".", "how're", "you", "abcdef"]
         
        self.assertEqual(tokens, expected) 
    
    def test_get_spatial_adjacencies_basic(self):
        """
        Test basic spatial adjacency detection in images.
        
        Creates a simple 2x2 image with 4 different colors and verifies:
        - All colors are detected
        - Adjacency counts are correct
        - Colors are sorted by frequency
        """
     
        quantized_image = np.array([
            [[0, 0, 0], [1, 1, 1]],
            [[2, 2, 2], [3, 3, 3]],
        ], dtype=np.uint8)
         
        adjacency_counts, unique_colors, color_frequencies = get_spatial_adjacencies(quantized_image)
         
        self.assertEqual(len(unique_colors), 4) 
        self.assertEqual(set(color_frequencies.values()), {1})
         
        self.assertEqual(sum(adjacency_counts.values()), 8)
         
        c00 = (0, 0, 0)
        c01 = (1, 1, 1)
        c10 = (2, 2, 2)
        c11 = (3, 3, 3)
        # Expected neighbors:
        # c00 adjacent to c01 and c10
        self.assertGreater(adjacency_counts[(c00, c01)], 0)
        self.assertGreater(adjacency_counts[(c00, c10)], 0) 
        self.assertGreater(adjacency_counts[(c11, c01)], 0)
        self.assertGreater(adjacency_counts[(c11, c10)], 0)
 



def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUnifiedNetworks)
    
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
    # Only run tests when script is executed directly
    success = run_tests()
