"""
This module implements a neuro-symbolic AI system that combines:
- Computer Vision (CIFAR-100 object recognition)
- Natural Language Processing (Skip-gram word embeddings)
- Symbolic Planning (PDDL planning)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Dict, Tuple, Optional
from pathlib import Path
import warnings
import os

from constants import *
from skipgram_modelling import *
from evolutionary_insertion import *
from vision_model import *
from planner import *

# ============================================================================
# SECTION 1: CIFAR-100 SEMANTIC EXPANSION
# ============================================================================
 
def build_my_embeddings(checkpoint_path: str = "models/best_skipgram_523words.pth") -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load and return your trained Skip-gram embeddings.
    
    This function serves as the entry point for loading your final embedding model
    that contains all Visual Genome words AND all 100 CIFAR-100 classes.
    
    Args:
        checkpoint_path: Path to your saved model checkpoint
        
    Returns:
        vocab: Dictionary mapping words to indices {word: index}
        embeddings: Numpy array of shape (vocab_size, embedding_dim)
        
    Example:
        >>> vocab, embeddings = build_my_embeddings()
        >>> print(f"Vocabulary size: {len(vocab)}")
        >>> print(f"Embedding dimension: {embeddings.shape[1]}")
        >>> print(f"'airplane' index: {vocab.get('airplane', 'NOT FOUND')}")
    """
    
    if os.path.exists(checkpoint_path):    
        # 1. Load checkpoint file 
        print(f"Loading best model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
        # 2. Extract the vocabulary dictionary
        vocab = checkpoint['vocab']
            
        # 3. Extract the embedding matrix
        embeddings = checkpoint['embeddings'].numpy()
        print(f"✅ Loaded embeddings for {len(vocab):,} words with embedding dimension {embeddings.shape[1]}")
    else:
        print("❌ Error: No saved model found!")
        # Raise an error to stop execution if the model is missing
        raise FileNotFoundError(f"{checkpoint_path} not found. Please train the model first.")
    
    # 4. Ensure vocabulary contains all required words (Visual Genome + CIFAR-100)    
    assert len(vocab) == 523, "Vocabulary size must be 523"
    
    
    return vocab, embeddings


# ============================================================================
# SECTION 2: NEURO-SYMBOLIC AI - MULTI-MODAL PLANNING
# ============================================================================

def plan_generator(input_data: Union[torch.Tensor, str],
                initial_state: List[str],               
                goal_state: List[str],                  
                domain_file: str = "plannning_files/domain.pddl",
                skipgram_path: str = "models/best_skipgram_523words.pth",
                projection_path: str = "models/best_cifar100_projection.pth") -> Optional[List[str]]:
    """
    Main entry point for the neuro-symbolic planning system.
    
    This function implements the complete pipeline from perception to planning.
    
    Args:
        input_data: Either an image tensor OR object name string
        initial_state: List of predicates describing initial state                      
        goal_state: List of predicates describing goal state                   
        domain_file: Path to the PDDL domain file
        skipgram_path: Path to Skip-gram embeddings checkpoint
        projection_path: Path to CIFAR-100 projection model checkpoint
        
    Returns:
        A list of action strings representing the plan, 
            OR None if:
                - The object cannot be identified
                - No valid plan exists
                - ...
        
    Example:
        >>> image = # CIFAR-100 image
        >>> initial = ["on table"]
        >>> goal = ["in basket"]
        >>> plan = plan_generator(image, initial, goal, "domain.pddl")        
    """
    DOMAIN_FILE = domain_file
    PROBLEM_FILE = "plannning_files/problem.pddl"       
    TRAIN_CONFIG = {
        'proj_dim': 115,                        # Output dimension of the vision projection head
        'lr': 5e-3,                             # AdamW learning rate
        'weight_decay': 1e-3,                   # L2 regularization
        'temperature': 1.0,                     # Scaling factor for the InfoNCE logits (controls contrast sharpness)
        'epochs': 100,                          # Maximum number of epochs
        'patience': 10,                         # Early stopping patience based on validation similarity
        'batch_sizes': {'train': 512, 'eval': 256},
        'save_path': projection_path
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_obj_token = '?x'


    all_vocab_words, embeddings = build_my_embeddings(skipgram_path)
    word_to_idx, idx_to_word = create_mappings(all_vocab_words)
    embeddings = torch.from_numpy(embeddings)

    #### Also alter the nodes names on the embeddings files and cifar_projection model ### 
    cifar_vocab = cifar100_classes_snake_case
    class_words = [w for w in cifar_vocab if w in word_to_idx]
    cifar100_vocab = [w.replace('_', '-') for w in class_words]
    print(f"Found {len(class_words) :,} words in the CIFAR-100 vocabulary.")
    
    all_vocab_words = list(word_to_idx.keys())
    vecs = [torch.as_tensor(embeddings[word_to_idx[w]]) for w in all_vocab_words]
    all_text_emb = torch.stack(vecs).to(device).float()
    print(f"Embeddings matrix is of shape {all_text_emb.shape}\n") 


    # Step 1: Identify the object name
    if isinstance(input_data, str):
        obj_name = input_data.lower().replace('_', '-')
        if obj_name not in all_vocab_words:
            print(f"Object '{obj_name}' not found in vocabulary.")
    elif isinstance(input_data, torch.Tensor):
        img_data = load_image(input_data)
        
        # Load the best model and predict the object name
        pred = predict_object_name(img_data, TRAIN_CONFIG, all_vocab_words, all_text_emb, device)
        obj_name = pred.lower().replace('_', '-')
        assert obj_name in exp_full_vocab, f"Object '{obj_name}' not found in vocabulary."
    else:
        raise ValueError("Input data must be either a string or a torch.Tensor")

    # Update the initial and goal state with the identified object name
    initial_state = set([state.replace(image_obj_token, obj_name) for state in initial_state])
    goal_state = set([state.replace(image_obj_token, obj_name) for state in goal_state])
    print(initial_state, goal_state)
    
    # 2. Parse the Domain and Problem files
    print(f"\nLoading files...\n  - {DOMAIN_FILE}\n  - {PROBLEM_FILE}\n")
   
    actions = PDDLParser.parse_domain(DOMAIN_FILE)
    objs, init, goal = PDDLParser.parse_problem(PROBLEM_FILE)
    print(f"✓ Domain: {len(actions)} actions found.")

    cifar_count = len(objs.get('item', set()) - TOOLS)
    tool_count = len(objs.get('tool', set()))
    print(f"✓ Problem: {cifar_count} CIFAR items + {tool_count} Tools.")
    print(f"✓ Initial state: {len(init.predicates)} predicates.\n")

     
    # Step 3: PDDL planning process: Validation -> Problem Gen -> Search -> Trace.
    try:
        action_sequence = plan_wrapper(obj_name,
                initial_state,
                goal_state,
                PROBLEM_FILE,
                DOMAIN_FILE)
     
        print_plan_execution(action_sequence, obj_name, initial_state)
        return action_sequence
    except Exception as e:
        print(f"Error: {e}")
        return e
    
    
# ==================== HELPER FUNCTIONS ====================
def load_image(img_data):
    """
    Preprocess an input image into a standard tensor format for the vision model.

    This helper converts a NumPy array or torch Tensor into a 3-channel tensor of
    shape (3, 224, 224), ensuring channel-first layout and applying resizing.

    Args:
        img_data: Input image as either:
            - np.ndarray with shape (H, W, 3) or (3, H, W), or
            - torch.Tensor with shape (H, W, 3) or (3, H, W)

    Returns:
        img_tensor: A torch.Tensor of shape (3, 224, 224) suitable for the vision encoder.

    Raises:
        AssertionError: If img_data is not a NumPy array or torch.Tensor.
        ValueError: If the input does not represent a 3-channel image.
        AssertionError: If the final tensor is not exactly shape (3, 224, 224).

    Example:
        >>> img = np.random.rand(32, 32, 3).astype(np.float32)
        >>> x = load_image(img)
        >>> print(x.shape)
        torch.Size([3, 224, 224])
    """


    assert isinstance(img_data, np.ndarray) or isinstance(img_data, torch.Tensor), "Input data must be either a string or a torch.Tensor or a numpy array."

    # Trnasform to Tensor
    if isinstance(img_data, np.ndarray):
        img_tensor = torch.from_numpy(img_data)
    else:
        img_tensor = img_data
    

    # Check if the image is already in the correct format
    if img_tensor.shape[0] == 3:
        print(f"Input image has shape {img_tensor.shape}")
    elif img_tensor.shape[2] == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    else:
        raise ValueError("Input data must be a 3-channel image of shape (3, H, W) or (H, W, 3).")

    # Resize and normalize the image
    transform = transforms.Compose([
                transforms.Resize( (224,224) ),
                # transforms.Normalize(
                #     mean = [0.485, 0.456, 0.406],
                #     std  = [0.229, 0.224, 0.225],
                # )
            ])
    img_tensor = transform(img_tensor)
    
    assert img_tensor.shape == (3, 224, 224), f"Input data must be a 3-channel image of shape (3, 224, 224). \nGot: {img_tensor.shape} instead.\n"

    return img_tensor

def predict_object_name(img_data: torch.Tensor,
                        TRAIN_CONFIG: Dict,
                        all_vocab_words: List[str],
                        all_text_emb, device: str):
    """
    Predict the most likely object name for an input image using cross-modal similarity.

    This function loads the trained image projection model from a checkpoint, encodes
    the input image into a projected embedding space, and selects the closest text
    embedding (cosine similarity) from the provided vocabulary embedding matrix.

    Args:
        img_data: A torch.Tensor image of shape (3, 224, 224).
        TRAIN_CONFIG: Dictionary containing model configuration, including:
            - 'proj_dim': projection dimension
            - 'save_path': path to the saved projection checkpoint
        all_vocab_words: List of vocabulary tokens aligned with all_text_emb rows.
        all_text_emb: Tensor of text embeddings of shape (vocab_size, proj_dim).
        device: Device string, e.g. "cpu" or "cuda".

    Returns:
        obj_name: Predicted object token as a string (with underscores replaced by hyphens).

    Raises:
        AssertionError: If img_data is not a torch.Tensor or has the wrong shape.
        FileNotFoundError: If the model checkpoint at TRAIN_CONFIG['save_path'] is missing.

    Example:
        >>> x = torch.randn(3, 224, 224)
        >>> name = predict_object_name(x, TRAIN_CONFIG, all_vocab_words, all_text_emb, "cpu")
        >>> print(name)
        'apple'
    """ 
    assert isinstance(img_data, torch.Tensor), "Input data must be a torch.Tensor."
    assert img_data.shape == (3, 224, 224), "Input data must be a 3-channel image of shape (3, 224, 224)."


    print("\n[Loading Best Model Image classfication model...]") 
    vision_model = ImageEncoder(proj_dim=TRAIN_CONFIG['proj_dim'], device=device)
    if not os.path.exists(TRAIN_CONFIG['save_path']):
        raise FileNotFoundError(f"Model checkpoint not found at {TRAIN_CONFIG['save_path']}. "
                            "Training must complete at least one epoch successfully.")

    # Load the saved parameters and move the model to evaluation mode
    checkpoint = torch.load(TRAIN_CONFIG['save_path'])
    vision_model.load_state_dict(checkpoint['model_state_dict'])


    vision_model.eval()
    with torch.no_grad():
        img = img_data.unsqueeze(0).to(device)  # add batch dimension
        _, visual_proj = vision_model(img)
        # normalize to use cosine similarity via dot product
        v = F.normalize(visual_proj, p=2, dim=1)                 # (1, D)
        t = F.normalize(all_text_emb.to(device), p=2, dim=1)         # (num_classes, D)

        sims = (v @ t.T).squeeze(0)                              # (num_classes,)
        pred_idx = int(torch.argmax(sims).item())                # index into class_words/text_emb
        obj_name = all_vocab_words[pred_idx]
    print("Predicted class of the input image is:", obj_name)   
    
    return obj_name.replace('_', '-')
 
def plan_wrapper(obj_name,
                 initial_state,
                 goal_state,
                 PROBLEM_FILE,
                 DOMAIN_FILE):
    """
    Run the full PDDL planning pipeline and return an execution trace.

    This helper wraps the symbolic planning workflow:
    1) Validate user-provided predicates (fail fast on typos)
    2) Generate a temporary problem file with the given initial/goal conditions
    3) Parse domain/problem and ground actions
    4) Filter actions to relevant objects/locations
    5) Run A* search and build a step-by-step trace of state changes
    6) Cleanup temporary files

    Args:
        obj_name: The grounded object name used to fill placeholders in the problem.
        initial_state: Iterable of predicates describing the starting world state.
        goal_state: Iterable of predicates describing the desired goal state.
        PROBLEM_FILE: Path to the base PDDL problem template file.
        DOMAIN_FILE: Path to the PDDL domain file.

    Returns:
        changes: A list of dictionaries (one per action step), each containing:
            - 'step': step index (1-based)
            - 'action': string form of the grounded action
            - 'added': sorted list of added predicates
            - 'removed': sorted list of removed predicates
        If no plan exists, returns None.

    Example:
        >>> init = {"at apple table", "hand-empty"}
        >>> goal = {"in apple basket"}
        >>> trace = plan_wrapper("apple", init, goal, "problem.pddl", "domain.pddl")
        >>> print(trace[0]["action"])
        pick-up apple table
    """


    # Ensure inputs are sets
    initial_state = set(initial_state)
    goal_state = set(goal_state)

    # 1. Validate Inputs (Fail fast if there are typos)
    validate_user_conditions(initial_state)
    validate_user_conditions(goal_state)

    # 2. Create Temporary Problem File
    # We pass the user's specific initial/goal states to override the defaults
    prob_defn = create_custom_problem(PROBLEM_FILE, initial_state, goal_state, obj_name)

    try:
        # 3. Parse Domain and the new Problem file
        actions_dict = PDDLParser.parse_domain(DOMAIN_FILE)
        objs, init, goal = PDDLParser.parse_problem(prob_defn)

        # 4. Ground Actions (Generate specific actions like 'pick-up apple')
        grounder = ActionGrounder(actions_dict, objs)
        grounded = grounder.ground_all()

        # 5. Run Search
        relevant_items, relevant_locs = extract_relevant_objects(init, goal)
        filtered_actions = filter_actions_by_relevance(grounded, relevant_items, relevant_locs)
        print("Started A* search...")
        plan = astar_search(
            initial=init,
            goal=goal,
            actions=filtered_actions,
            heuristic=h_combined
        )
        print("Finished A* search.")
        if plan is None:
            return None
        
        # 6. Build Execution Trace
        changes, curr_state = [], init
        for i, action in enumerate(plan, 1):
            curr_state = curr_state.apply_action(action)
            changes.append({
                'step': i,
                'action': str(action),
                'added': sorted(map(str, action.add_effects)),
                'removed': sorted(map(str, action.del_effects))
            })

        return changes
    
    finally:
        # 7. Cleanup (Delete temporary file)
        if os.path.exists(prob_defn):
            os.unlink(prob_defn)
