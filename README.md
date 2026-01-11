# Neuro-Symbolic-AI
This project builds a neuro-symbolic planning pipeline that links perception to symbolic reasoning. It identifies an object from either an image (via a projection model matched against word embeddings) or a text label, then uses the provided PDDL domain to construct a planning problem and search for a valid action sequence that achieves the goal.


A custom implementation of a **neuro‑symbolic pipeline** from scratch, that links:

- **Vision** (CIFAR‑100 image recognition)
- **Language** (Skip‑gram modelling)
- **Symbolic Planning** (PDDL + A* search)

The main entry points are in **`main.py`**:

- `build_my_embeddings(...)` – loads the final Skip‑gram embedding matrix (Visual Genome words + CIFAR‑100 classes).
- `plan_generator(...)` – end‑to‑end pipeline: image/object → grounded PDDL problem → plan.


## How it works (high level)

1. **Perception**
   - If `input_data` is a **string**, it is treated as the target object name.
   - If `input_data` is an **image tensor**, the image is resized to `224×224` and passed through a pretrained
     MobileNetV3 encoder + projection head (see `vision_model.py`). The output is matched against the full text embedding
     matrix using cosine similarity.

2. **Symbol mapping**
   - The predicted token is normalized to **kebab‑case** (e.g., `sweet-pepper`, `pickup-truck`) to match the PDDL
     vocabulary.

3. **Planning**
   - The user-provided `initial_state` / `goal_state` strings may include the placeholder `?x`.
   - A temporary problem file is created from `problem.pddl` with `?x` replaced by the identified object.
   - The PDDL domain/problem are parsed, actions are grounded, filtered to relevant objects, and an **A*** search
     is run (see `planner.py`).
   - The returned plan is converted into an execution trace (added/removed predicates per step).

## Project layout

```
├─ main.py                      # Coursework entry point (required function signatures)
├─ constants.py                 # CIFAR vocab + helper constants
├─ network_processing.py        # Text/network utilities (used by SGNS pipeline)
├─ skipgram_modelling.py        # Skip-gram with negative sampling (SGNS)
├─ evolutionary_insertion.py    # Evolutionary strategy for embedding insertion
├─ vision_model.py              # Vision encoder + projection head (MobileNetV3)
├─ planner.py                   # PDDL parser, grounding, and search (A*)
├─ README.md                    # Project documentation / usage
└─ plannning_files/
    ├─ domain.pddl                          # Planning domain
    └─ problem.pddl                         # Base problem definition   
└─ models/
    ├─ best_skipgram_523words.pth           # Trained Skip-gram checkpoint  
    └─ best_cifar100_projection.pth         # Trained vision→embedding projection checkpoint

```

## Requirements

Python 3.9+ (recommended).

Key Python dependencies used across:

- `torch`, `torchvision`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `networkx`
- `requests`
- `Pillow`

A minimal install (CPU) typically looks like:

```bash
pip install torch torchvision
pip install numpy scikit-learn matplotlib tqdm networkx requests pillow
```

> GPU is optional. If CUDA is available, the vision model will run on GPU automatically.

## Quick start

Run from the directory containing `main.py` (so the default relative paths to the `.pth` and `.pddl` files resolve correctly).

### 1) Load embeddings

```python
from main import build_my_embeddings

vocab, embeddings = build_my_embeddings("best_skipgram_523words.pth")
print(len(vocab), embeddings.shape)
```

### 2) Generate a plan (string input)

`plan_generator` accepts **either** a string object name **or** an image tensor.

- CIFAR object names are expected in **kebab‑case** (e.g., `aquarium-fish`, `pickup-truck`).
- The planner predicates must be valid for the provided PDDL domain (see `domain.pddl`).

```python
from main import plan_generator

initial_state = [
    "(agent-at lab)",
    "(at ?x lab)",
    "(whole ?x)",
    "(clear ?x)",
]

goal_state = [
    "(cut-into-pieces ?x)",
]

plan = plan_generator(
    input_data="apple",          # or "aquarium-fish", "pickup-truck", ...
    initial_state=initial_state,
    goal_state=goal_state,
    domain_file="domain.pddl",
)

print(plan)
```

### 3) Generate a plan (image input)

When `input_data` is a tensor, it should represent a **3‑channel image**.
The helper `load_image(...)` in `main.py` will try to coerce `(H, W, 3)` to `(3, H, W)` and resize to `(3, 224, 224)`.

```python
import torch
from main import plan_generator

# Example dummy tensor (replace with a real CIFAR‑100 image tensor)
img = torch.rand(3, 32, 32)

initial_state = [
    "(agent-at lab)",
    "(at ?x lab)",
    "(whole ?x)",
    "(clear ?x)",
]

goal_state = [
    "(documented ?x)",
]

plan = plan_generator(img, initial_state, goal_state)
print(plan)
```

## What `plan_generator` returns

On success, the function returns a **list of step dictionaries** (execution trace). Each step includes:

- `step` – step number (1‑indexed)
- `action` – the grounded action string, e.g. `(pick-up knife lab)`
- `added` / `removed` – predicates added/removed by that action

If planning fails, it returns `None` (no plan) or an exception object (caught and printed by the wrapper).

## Notes & common pitfalls
 
- **Predicate syntax:** user‑provided predicates must include parentheses and only use valid arguments (CIFAR items, tools, or locations). See `planner.validate_user_conditions`
- **Hyphen vs underscore:** CIFAR class names are normalized to kebab‑case in planning (e.g., `sweet-pepper`).
 