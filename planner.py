import re
import tempfile
from typing import Set, List, Dict, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import heapq
from typing import Callable

# ==================== CONSTANTS ====================
CIFAR_100_CLASSES = {
    "apple", "aquarium-fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn-mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple", "motorcycle", "mountain", "mouse",
    "mushroom", "oak", "orange", "orchid", "otter", "palm", "pear",
    "pickup-truck", "pine", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet-pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow", "wolf", "woman", "worm"
}

TOOLS = {'knife', 'dslr'}
LOCATIONS = {'lab', 'outdoors'}
IGNORED_KEYWORDS = {'item', 'location', 'object', 'objects', '-',
                'agent', 'define', 'problem', 'domain', 'init', 'goal'}

# ==================== DATA STRUCTURES ====================

@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[str, ...]
    
    def __str__(self) -> str:
        return f"({self.name} {' '.join(self.args)})" if self.args else f"({self.name})"
    
    @staticmethod
    def from_string(s: str) -> 'Predicate':
        parts = s.strip().strip('()').split()
        return Predicate(parts[0], tuple(parts[1:]) if len(parts) > 1 else ())


@dataclass(frozen=True)
class Action:
    name: str
    parameters: Tuple[str, ...]
    preconditions: FrozenSet[Predicate]
    add_effects: FrozenSet[Predicate]
    del_effects: FrozenSet[Predicate]

    def __str__(self) -> str:
        return f"({self.name} {' '.join(self.parameters)})"
    
    def instantiate(self, bindings: Dict[str, str]) -> 'Action':
        """
        Creates a concrete Action instance by replacing variables with objects.
        bindings example: {'?x': 'apple', '?loc': 'lab'}
        """ 
        sub = lambda pred: Predicate(pred.name, tuple(bindings.get(arg, arg) for arg in pred.args))
        return Action(name=self.name, 
               parameters=tuple(bindings.get(p,p) for p in self.parameters),
               preconditions=frozenset( sub(pred) for pred in self.preconditions),
               add_effects=frozenset(  sub(pred) for pred in self.add_effects),
               del_effects=frozenset(   sub(pred) for pred in self.del_effects))
     

@dataclass(frozen=True)
class State:
    predicates: FrozenSet[Predicate]
    
    def apply_action(self, action: Action) -> 'State':
        """
        Returns a NEW State by applying the action's effects.
        Recall: Next State = (Current State - Del Effects) + Add Effects
        """ 
        return State( (self.predicates - action.del_effects) | action.add_effects  )
    


    def is_applicable(self, action: Action) -> bool:
        """
        Checks if an action can be applied to this state.
        Recall: All preconditions must exist in the current state.
        """ 
        return action.preconditions.issubset(self.predicates)
        
    def satisfies(self, goal: FrozenSet[Predicate]) -> bool:
        return goal.issubset(self.predicates)


@dataclass(order=True)
class SearchNode:
    f_score: int
    state: State = field(compare=False)
    action: Optional[Action] = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False)
    g_score: int = field(compare=False)
    
    def get_plan(self) -> List[Action]:
        plan, node = [], self
        while node.parent:
            plan.append(node.action)
            node = node.parent
        return list(reversed(plan))

# ==================== PDDL PARSER ====================

class PDDLParser:
    last_discarded = set()
    
    @staticmethod
    def _extract_predicates(block: str) -> Set[Predicate]:
        """Extract predicates from a block, handling 'and' wrappers and nested parens."""
        block = block.strip()
        if block.startswith('(and'):
            depth, i = 0, 4
            while i < len(block):
                if block[i] == '(':
                    depth += 1
                elif block[i] == ')':
                    depth -= 1
                    if depth == -1:
                        block = block[4:i].strip()
                        break
                i += 1
        
        preds, depth, curr = set(), 0, []
        for c in block:
            if c == '(':
                depth += 1
                curr.append(c)
            elif c == ')':
                depth -= 1
                curr.append(c)
                if depth == 0 and curr:
                    s = ''.join(curr).strip()
                    if s and not s.startswith('(not') and not s.startswith('(forall'):
                        try:
                            preds.add(Predicate.from_string(s))
                        except:
                            pass
                    curr = []
            elif depth > 0:
                curr.append(c)
        return preds
    
    @staticmethod
    def _parse_effects(body: str) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Parse add and delete effects from action body."""
        m = re.search(r':effect\s+(\(.*)', body, re.DOTALL | re.IGNORECASE)
        if not m:
            return set(), set()
        
        effect_block = m.group(1).strip()
        depth, i = 0, 0
        while i < len(effect_block):
            if effect_block[i] == '(':
                depth += 1
            elif effect_block[i] == ')':
                depth -= 1
                if depth == 0:
                    effect_block = effect_block[:i+1]
                    break
            i += 1
        
        block = effect_block[4:].strip().rstrip(')') if effect_block.startswith('(and') else effect_block
        adds, dels, depth, curr = set(), set(), 0, []
        
        for c in block:
            if c == '(':
                depth += 1
                curr.append(c)
            elif c == ')':
                depth -= 1
                curr.append(c)
                if depth == 0 and curr:
                    s = ''.join(curr).strip()
                    if s.startswith('(not'):
                        inner = re.search(r'\(not\s+(\(.*?\))\s*\)', s, re.IGNORECASE)
                        if inner:
                            try:
                                dels.add(Predicate.from_string(inner.group(1)))
                            except:
                                pass
                    else:
                        try:
                            adds.add(Predicate.from_string(s))
                        except:
                            pass
                    curr = []
            elif depth > 0:
                curr.append(c)
        
        return adds, dels
    
    @staticmethod
    def parse_domain(path: str) -> Dict[str, Action]:
        """Parse domain file and return action schemas."""
        with open(path) as f:
            content = f.read()
        
        actions = {}
        for m in re.finditer(r'\(:action\s+(\S+)(.*?)(?=\(:action|\Z)', content, re.DOTALL | re.IGNORECASE):
            name, body = m.groups()
            
            params = []
            if params_match := re.search(r':parameters\s+\((.*?)\)', body, re.DOTALL | re.IGNORECASE):
                params = re.findall(r'\?[\w-]+', params_match.group(1))
            
            preconds = set()
            if precond_match := re.search(r':precondition\s+(\(.*?)(?=\s*:effect|\Z)', body, re.DOTALL | re.IGNORECASE):
                preconds = PDDLParser._extract_predicates(precond_match.group(1))
            
            adds, dels = PDDLParser._parse_effects(body)
            actions[name] = Action(name, tuple(params), frozenset(preconds), frozenset(adds), frozenset(dels))
        
        return actions
    
    @staticmethod
    def parse_problem(path: str) -> Tuple[Dict[str, Set[str]], State, FrozenSet[Predicate]]:
        """Parse problem file and return objects, initial state, and goal."""
        with open(path) as f:
            content = f.read()
        
        objs = defaultdict(set)
        PDDLParser.last_discarded = set()
        
        if om := re.search(r':objects(.*?)(?=\(:init)', content, re.DOTALL | re.IGNORECASE):
            ob = re.sub(r';.*$', '', om.group(1), flags=re.MULTILINE).strip()
            for token in re.findall(r'[^\s():;]+', ob):
                t_lower = token.lower()
                if t_lower in CIFAR_100_CLASSES:
                    objs['item'].add(t_lower)
                elif t_lower in TOOLS:
                    objs['item'].add(t_lower)
                    objs['tool'].add(t_lower)
                elif t_lower in LOCATIONS:
                    objs['location'].add(t_lower)
                elif t_lower not in IGNORED_KEYWORDS:
                    PDDLParser.last_discarded.add(t_lower)
        
        for t in TOOLS:
            objs['item'].add(t)
            objs['tool'].add(t)
        
        init = set()
        if im := re.search(r':init(.*?)(?=\(:goal|\Z)', content, re.DOTALL | re.IGNORECASE):
            init = PDDLParser._extract_predicates(im.group(1))
        
        goal = set()
        if gm := re.search(r':goal\s+(\(.*\))', content, re.DOTALL | re.IGNORECASE):
            goal = PDDLParser._extract_predicates(gm.group(1))
        
        return dict(objs), State(frozenset(init)), frozenset(goal)


# ==================== ACTION GROUNDING ====================
class ActionGrounder:
    def __init__(self, schemas: Dict[str, Action], objects: Dict[str, Set[str]]):
        self.schemas = schemas
        self.all_items = sorted(objects.get('item', set()))
        self.cifar_objects = sorted(set(self.all_items) - TOOLS)
        self.locations = sorted(LOCATIONS)
    
    def ground_all(self) -> List[Action]:
        """Ground all action schemas with concrete objects."""
        grounded = []
        
        for name, schema in self.schemas.items():
            if name == 'walk-between-rooms':
                grounded.extend(self._ground_walk(schema))
            elif name in ['pick-up', 'put-down']:
                grounded.extend(self._ground_item_location(schema))
            elif name in ['stack', 'unstack']:
                grounded.extend(self._ground_stack(schema))
            elif name in ['slice-object', 'clean-object', 'take-photo']:
                grounded.extend(self._ground_object_action(schema))
        
        return grounded
    
    def _ground_walk(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: l1, schema.parameters[1]: l2})
                for l1 in self.locations for l2 in self.locations if l1 != l2]
    
    def _ground_item_location(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: item, schema.parameters[1]: loc})
                for item in self.all_items for loc in self.locations]
    
    def _ground_stack(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: top, schema.parameters[1]: bottom, schema.parameters[2]: loc})
                for top in self.all_items for bottom in self.all_items 
                if top != bottom for loc in self.locations]
    
    def _ground_object_action(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: obj, schema.parameters[1]: loc})
                for obj in self.cifar_objects for loc in self.locations]



# ==================== BFS PLANNER ====================

def bfs_search(initial: State, 
               goal: FrozenSet[Predicate],
               actions: List[Action], 
               max_iter: int = 50000,
               verbose: bool = True) -> Optional[List[Action]]:
    
    # 1. Check if we are already there
    if initial.satisfies(goal):
        return []
    
    # 2. Setup Frontier and Visited set
    start_node = SearchNode(0, initial, None, None, 0)
    frontier = deque([start_node]) # FIFO Queue
    visited = {initial}
    
    iters = 0

    while frontier and iters < max_iter:
        iters += 1
        if verbose and iters % 1000 == 0:
            print(f"Iter {iters:,} | Frontier: {len(frontier):,} | Visited: {len(visited):,}")
 
        # 1. Pop the next node (FIFO)
        node = frontier.popleft()

        # 2. Check for goal
        if node.state.satisfies(goal):
            if verbose:
                print(f"✓ Solution found in {iters} iterations. Plan length: {node.g_score}")
            return node.get_plan()

        # 3. Generate successors (apply all applicable grounded actions)
        for act in actions:
            if not node.state.is_applicable(act):
                continue

            next_state = node.state.apply_action(act)

            if next_state in visited:
                continue

            visited.add(next_state)

            next_g = node.g_score + 1  # BFS depth
            child = SearchNode(
                f_score=next_g,          # for BFS, f = g
                state=next_state,
                action=act,
                parent=node,
                g_score=next_g
            )
            frontier.append(child)


    print(f"✗ No solution. Iterations: {iters}")
    return None


##### HELPERS ######
def _get_agent_loc(state: State) -> Optional[str]:
    for p in state.predicates:
        if p.name == 'agent-at' and len(p.args) == 1:
            return p.args[0]
    return None

def _get_obj_loc(state: State, obj: str) -> Optional[str]:
    for p in state.predicates:
        if p.name == 'at' and len(p.args) == 2 and p.args[0] == obj:
            return p.args[1]
    return None

def _is_holding(state: State, obj: str) -> bool:
    return Predicate('holding', (obj,)) in state.predicates

def _move_cost(loc1: Optional[str], loc2: Optional[str]) -> int:
    if loc1 is None or loc2 is None:
        return 0  # keep heuristic admissible (don’t overestimate)
    return 0 if loc1 == loc2 else 1  # in this domain, rooms are directly connected
def _get_agent_loc(state: State) -> Optional[str]:
    for p in state.predicates:
        if p.name == 'agent-at' and len(p.args) == 1:
            return p.args[0]
    return None

def _get_obj_loc(state: State, obj: str) -> Optional[str]:
    for p in state.predicates:
        if p.name == 'at' and len(p.args) == 2 and p.args[0] == obj:
            return p.args[1]
    return None

def _is_holding(state: State, obj: str) -> bool:
    return Predicate('holding', (obj,)) in state.predicates

def _move_cost(loc1: Optional[str], loc2: Optional[str]) -> int:
    if loc1 is None or loc2 is None:
        return 0  # keep heuristic admissible (don’t overestimate)
    return 0 if loc1 == loc2 else 1  # in this domain, rooms are directly connected
 
def build_action_index(actions: List[Action]) -> Dict[Predicate, List[Action]]:
    idx = defaultdict(list)
    for a in actions:
        for p in a.preconditions:
            idx[p].append(a)
    return idx

def candidate_actions(state: State, index: Dict[Predicate, List[Action]]) -> Set[Action]:
    cands = set()
    for p in state.predicates:
        cands.update(index.get(p, ()))
    return cands


# ----- HEURISTICS    -----
def h_unsatisfied_goals(state: State, goal: FrozenSet[Predicate], actions: List[Action]) -> int:
    """Heuristic = number of goal predicates not yet true (fast, but not always admissible)."""
    return len(goal - state.predicates)

def h_goal_cover_lower_bound(state: State, goal: FrozenSet[Predicate], actions: List[Action]) -> int:
    """
    Lower-bound style heuristic:
      h = ceil(#unsatisfied_goals / max_goal_add_per_action)
    This is usually more conservative.
    """
    unsat = len(goal - state.predicates)
    if unsat == 0:
        return 0

    max_add = 0
    for a in actions:
        max_add = max(max_add, len(a.add_effects & goal))
    max_add = max(1, max_add)  # avoid divide by 0

    return int(math.ceil(unsat / max_add))
 
def h_transport_at_goal(state: State, goal: FrozenSet[Predicate], actions: List[Action]) -> int:
    """
    Domain-aware admissible-ish heuristic for goals like (at obj loc).
    Returns a lower bound on remaining steps.
    """
    # If multiple goals, take max (still a lower bound-ish strategy)
    best = 0

    agent_loc = _get_agent_loc(state)

    for g in goal:
        if g.name != 'at' or len(g.args) != 2:
            continue
        obj, goal_loc = g.args

        # already satisfied
        if g in state.predicates:
            continue

        if _is_holding(state, obj):
            # need to move to goal_loc + put-down
            cost = _move_cost(agent_loc, goal_loc) + 1
        else:
            obj_loc = _get_obj_loc(state, obj)
            # move to object + pick-up + move to goal + put-down
            cost = _move_cost(agent_loc, obj_loc) + 1 + _move_cost(obj_loc, goal_loc) + 1

        best = max(best, cost)

    return best

def h_documented(state: State, goal: FrozenSet[Predicate], actions) -> int:
    # lower bound for documented(obj): need to be at obj location + take-photo
    agent_loc = _get_agent_loc(state)
    best = 0
    for g in goal:
        if g.name == "documented" and len(g.args) == 1:
            obj = g.args[0]
            if g in state.predicates:
                continue
            obj_loc = _get_obj_loc(state, obj)
            # must be at object's location (or have it there already) then take-photo
            cost = _move_cost(agent_loc, obj_loc) + 1
            best = max(best, cost)
    return best

def h_combined(state, goal, actions):
    return max(h_transport_at_goal(state, goal, actions),
               h_documented(state, goal, actions))


       


# ----- A* search    -----
def astar_search(
    initial: State,
    goal: FrozenSet[Predicate],
    actions: List[Action],
    heuristic: Callable[[State, FrozenSet[Predicate], List[Action]], int] = h_unsatisfied_goals,
    max_iter: int = 200000,
    verbose: bool = True
) -> Optional[List[Action]]:

    # 1) Goal already satisfied?
    if initial.satisfies(goal):
        return []
    
    # 2) Priority queue of (f_score, tie_breaker, SearchNode)
    # tie_breaker avoids comparing nodes when f ties
    open_heap: List[Tuple[int, int, SearchNode]] = []
    tie = 0

    h0 = heuristic(initial, goal, actions)
    start = SearchNode(
        f_score=h0,
        state=initial,
        action=None,
        parent=None,
        g_score=0
    )
    heapq.heappush(open_heap, (start.f_score, tie, start))

    # 3) Best g-score seen for each state (A* needs this, not just a visited set)
    best_g: Dict[State, int] = {initial: 0}
    iters = 0
    while open_heap and iters < max_iter:
        iters += 1
        _, _, node = heapq.heappop(open_heap)

        # Skip stale heap entries (we found a better path to this state already)
        if node.g_score != best_g.get(node.state, float("inf")):
            continue
        # Goal test
        if node.state.satisfies(goal):
            if verbose:
                print(f"✓ Solution found in {iters} iterations. Cost(g): {node.g_score}, f: {node.f_score}")
            return node.get_plan()
        # Expand
        for act in actions:
            if not node.state.is_applicable(act):
                continue

            next_state = node.state.apply_action(act)
            next_g = node.g_score + 1

            # Only keep if this is the best path to next_state so far
            if next_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = next_g
                h = heuristic(next_state, goal, actions)
                f = next_g + h

                tie += 1
                child = SearchNode(
                    f_score=f,
                    state=next_state,
                    action=act,
                    parent=node,
                    g_score=next_g
                )
                heapq.heappush(open_heap, (f, tie, child))

        if verbose and iters % 1000 == 0:
            print(f"Iter {iters:,} | Open: {len(open_heap):,} | Best_g states: {len(best_g):,}")

    if verbose:
        print(f"✗ No solution. Iterations: {iters}")
    return None


def extract_relevant_objects(initial: State, goal: FrozenSet[Predicate]) -> Tuple[Set[str], Set[str]]:
    items, locs = set(), set()
    for pred in list(initial.predicates) + list(goal):
        for arg in pred.args:
            if arg in LOCATIONS:
                locs.add(arg)
            elif arg in CIFAR_100_CLASSES or arg in TOOLS:
                items.add(arg)
    # If goal/init doesn't mention locations explicitly, keep all.
    if not locs:
        locs = set(LOCATIONS)
    return items, locs


def filter_actions_by_relevance(actions: List[Action], relevant_items: Set[str], relevant_locs: Set[str]) -> List[Action]:
    """
    Keep:
      - all walk actions among relevant locations
      - other actions only if they mention a relevant item (object/tool)
    """
    kept = []
    for a in actions:
        params = set(a.parameters)
        if a.name == 'walk-between-rooms':
            # only keep moves among locations we care about
            if params.issubset(relevant_locs):
                kept.append(a)
        else:
            # keep if action touches relevant items (e.g., dinosaur)
            if params & relevant_items:
                kept.append(a)
    return kept 



# ==================== PROBLEM GENERATION ====================

def create_custom_problem(base_path: str, init_overrides: Set[str], 
                         goals: Set[str], name: str = "custom") -> str:
    """Generate custom problem file with conflict resolution."""
    with open(base_path) as f:
        content = f.read()
    
    objs_match = (re.search(r'(\(:objects.*?\)(?=\s*\(:init))', content, re.DOTALL | re.IGNORECASE) or
                  re.search(r'(\(:objects.*?\))', content, re.DOTALL | re.IGNORECASE))
    objs_sec = objs_match.group(1) if objs_match else "(:objects)"
    
    base_init = set()
    if base_init_match := re.search(r':init(.*?)(?=\(:goal|\Z)', content, re.DOTALL | re.IGNORECASE):
        base_init = PDDLParser._extract_predicates(base_init_match.group(1))
    
    user_init = {Predicate.from_string(p) for p in init_overrides}
    user_goal = {Predicate.from_string(p) for p in goals}
    
    # Build conflict map
    user_defs = defaultdict(set)
    for p in user_init:
        if p.args:
            user_defs[p.args[0]].add(p.name)
        user_defs['GLOBAL'].add(p.name)
    
    # Resolve conflicts
    final_init = set()
    for bp in base_init:
        if bp in user_init:
            continue
        
        conflict = _has_conflict(bp, user_defs)
        if not conflict:
            final_init.add(bp)
    
    final_init.update(user_init)
    
    problem_str = f"""(define (problem {name}-problem)
  (:domain cifar100-process)
  {objs_sec}
  (:init {chr(10).join(f"    {p}" for p in sorted(map(str, final_init)))} )
  (:goal (and {chr(10).join(f"      {p}" for p in sorted(map(str, user_goal)))} ))
)"""
    
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False)
    tf.write(problem_str)
    tf.close()
    return tf.name


def _has_conflict(pred: Predicate, user_defs: Dict[str, Set[str]]) -> bool:
    """Check if base predicate conflicts with user definitions."""
    global_defs = user_defs['GLOBAL']
    
    if pred.name == 'hand-empty' and 'holding' in global_defs:
        return True
    if pred.name == 'holding' and 'hand-empty' in global_defs:
        return True
    if pred.name == 'agent-at' and 'agent-at' in global_defs:
        return True
    
    if pred.args:
        obj_defs = user_defs[pred.args[0]]
        if pred.name in ['at', 'on-top'] and {'at', 'on-top'} & obj_defs:
            return True
        if pred.name == 'clear' and {'at', 'on-top', 'clear'} & obj_defs:
            return True
        if pred.name == 'whole' and 'cut-into-pieces' in obj_defs:
            return True
        if pred.name == 'wet' and 'clean' in obj_defs:
            return True
        if pred.name == 'clean' and 'wet' in obj_defs:
            return True
    
    return False

# ==================== VALIDATION ====================

def validate_user_conditions(conditions: Set[str]):
    """Validate user input for syntax and vocabulary."""
    valid_vocab = CIFAR_100_CLASSES | TOOLS | LOCATIONS
    
    for s in conditions:
        s_clean = s.strip()
        if not (s_clean.startswith('(') and s_clean.endswith(')')):
            raise ValueError(f"SYNTAX ERROR: '{s}' is missing parentheses.")
        
        pred = Predicate.from_string(s_clean)
        for arg in pred.args:
            if arg not in valid_vocab:
                raise ValueError(
                    f"VOCABULARY ERROR: Unknown object '{arg}' in predicate '{s}'.\n"
                    f"   (Check spelling. Valid args are 100 CIFAR items, 2 tools, or 'lab'/'outdoors')"
                )

# ==================== VISUALIZATION ====================

def print_plan_execution(plan: List[Dict], object_name: str, initial_conditions: Set[str]):
    """Print formatted execution trace."""
    print("\n" + "="*80)
    print(f"📋 EXECUTION TRACE: {object_name.upper()}")
    print("="*80)
    print(f"{'STEP':<6} | {'ACTION':<35} | {'STATE CHANGES / INITIAL STATE':<40}")
    print("-" * 90)
    
    init_str = ", ".join(sorted(initial_conditions))
    print(f"{'0':<6} | {'(INITIAL STATE)':<35} | {init_str}")
    
    if not plan:
        print("-" * 90)
        print("✓ GOAL ALREADY ACHIEVED (No actions needed)")
        print("="*80 + "\n")
        return
    
    for step in plan:
        changes = []
        if step['added']:
            changes.append(f"++ {', '.join(step['added'])}")
        if step['removed']:
            changes.append(f"-- {', '.join(step['removed'])}")
        changes_str = " | ".join(changes)
        
        print(f"{step['step']:<6} | {step['action']:<35} | {changes_str}")
    
    print("-" * 90)
    print("✓ GOAL ACHIEVED")
    print("="*80 + "\n")
