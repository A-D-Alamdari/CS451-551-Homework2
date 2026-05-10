# env/ — Mars Rover Environments

## Purpose

This package contains all environment logic for the Mars Rover RL
project: the abstract MDP and Environment interfaces, the concrete
Mars Grid World MDP (including `.lay` file loading), and the
Crawler robot domain. Nothing in this package touches student code
or grading — it is pure infrastructure that both students and the
autograder consume.

---

## Files

### mdp.py — Abstract Markov Decision Process

Defines the `MarkovDecisionProcess` base class whose six methods
describe a planning problem in full. Every method raises
`NotImplementedError`; `MarsGrid` is the only concrete subclass.

| Method | Returns | Description |
|---|---|---|
| `get_states()` | `list[Any]` | Every state in the MDP (including `TERMINAL_STATE`) |
| `get_start_state()` | `Any` | The unique initial state s₀ |
| `get_possible_actions(state)` | `tuple[Any, ...]` | Legal actions from `state`; empty tuple for terminals |
| `get_transition_states_and_probs(state, action)` | `list[tuple[Any, float]]` | Stochastic T(s, a, ·) as `[(s', prob), ...]`, sums to 1.0 |
| `get_reward(state, action, next_state)` | `float` | Scalar reward R(s, a, s') |
| `is_terminal(state)` | `bool` | Whether `state` is absorbing (no actions, no further reward) |

**Used by:** `MarsGrid`, `ValueIterationAgent`, autograder test classes.

---

### environment.py — Abstract Interactive Environment

Defines the `Environment` base class for model-free learners. An
`Environment` wraps an MDP and stores the agent's current position
internally. Crucially, there is **no method for peeking at
transition probabilities** — a correct Q-learning agent must never
call one.

| Method | Returns | Description |
|---|---|---|
| `get_current_state()` | `Any` | Where the agent is right now |
| `get_possible_actions(state)` | `tuple[Any, ...]` | Legal moves from `state` |
| `do_action(action)` | `tuple[Any, float]` | Sample a successor, return `(next_state, reward)` |
| `reset()` | `None` | Start a fresh episode |
| `is_terminal()` | `bool` | Has the current episode ended? (takes no argument) |

**Used by:** `MarsGridEnvironment`, `CrawlerEnvironment`.

---

### mars_grid.py — The Core Mars Grid World MDP

The largest module in the package. Contains the concrete `MarsGrid`
MDP, its interactive wrapper `MarsGridEnvironment`, the `.lay` file
loader, and a registry of hardcoded fallback grid layouts.

#### Classes

**`MarsGrid(MarkovDecisionProcess)`** — a 2-D stochastic grid world.

- **Constructor:** `MarsGrid(grid_text, noise=0.2, living_reward=0.0, discount=0.9)`
- **Attributes:** `width`, `height`, `cells[x][y]`, `noise`, `living_reward`, `discount`, `start_state`

**`MarsGridEnvironment(Environment)`** — interactive wrapper.

- **Constructor:** `MarsGridEnvironment(grid_mdp)`
- **`do_action(action)`** samples from the MDP's transition probabilities using `util.sample`.

#### Grid cell types

| Token (in `.lay` / `grid_text`) | Internal value | Meaning |
|---|---|---|
| `_` or `.` or `' '` or `''` or `None` | `' '` | Open terrain — rover can pass through |
| `#` | `'#'` | Wall — impassable, blocks movement |
| `S` | `' '` + start recorded | Start position (also open terrain) |
| `int` (e.g. `1`, `-10`) | `float` (e.g. `1.0`, `-10.0`) | Terminal reward cell (positive = sample, negative = crater) |
| `float` (e.g. `3.5`) | `float` | Terminal reward cell |

Booleans are explicitly rejected (`ValueError`).

#### Coordinate system

- Coordinates are `(x, y)` where **x increases rightward** and **y
  increases upward**.
- `grid_text` is given **top-row-first** (like reading a page), so
  the constructor flips the y-axis: input row `i` maps to
  `y = height - 1 - i`.
- `cells[x][y]` stores the parsed cell at position `(x, y)` after
  the flip. `y = 0` is the **bottom** of the map.

#### Movement noise (solar interference)

When the rover attempts to move in a direction, the actual outcome
is stochastic:

| Outcome | Probability | Description |
|---|---|---|
| Intended direction | `1 - noise` (default 0.8) | The rover goes where it meant to |
| Left perpendicular | `noise / 2` (default 0.1) | Solar wind nudges the rover left |
| Right perpendicular | `noise / 2` (default 0.1) | Solar wind nudges the rover right |

If any outcome would move the rover into a wall or off the grid,
the rover **stays in place** — but that outcome's probability mass
still counts. Because multiple outcomes can collapse to the same
cell (e.g., slipping left into a wall and slipping right into a
wall both result in staying put), the probabilities are accumulated
in a `util.Counter` before being returned.

#### Terminal mechanics

- Cells with a numeric reward are *reward cells*. Their only legal
  action is `'extract'`.
- The `'extract'` action transitions **deterministically** to
  `'TERMINAL_STATE'` (a string sentinel) with probability 1.0.
- The reward for `'extract'` is the cell's numeric value (positive
  for sample sites, negative for craters).
- `'TERMINAL_STATE'` is the absorbing terminal: `is_terminal`
  returns `True`, `get_possible_actions` returns `()`,
  `get_reward` returns `0.0`.
- All other (non-extract) moves pay `living_reward` (default 0.0).

#### Key functions

| Function | Description |
|---|---|
| `load_grid_from_file(filepath)` | Parse a `.lay` file → `list[list]` of cell tokens |
| `get_layout_path(name)` | Resolve `layouts/<name>.lay` relative to the project root |
| `build_mars_grid(name, noise, living_reward, discount)` | Try `.lay` file first, fall back to `GRID_REGISTRY`, return a `MarsGrid` |
| `GRID_REGISTRY` | `dict[str, Callable]` — 6 hardcoded fallback layouts |

#### Layout file format (`.lay`)

```
// comment line (ignored)
_ _ _ -1
_ # _ _
S _ _ 1
```

- One row per line, tokens separated by whitespace.
- `_` or `.` = open, `#` = wall, `S` = start, numbers = terminals.
- `//` comments and blank lines are ignored (note: `#` is for walls,
  not comments).
- All rows must be the same width.

---

### crawler.py — Crawler Robot Domain

A two-segment arm crawler in a 2-D simulation chamber. This is the
second RL domain in the project — a continuing (non-episodic) task
that tests Q-learning generalisation.

| Aspect | Detail |
|---|---|
| **State** | `(arm_index, hand_index)` — integer indices into discretised angle arrays |
| **Angle range** | `[-π/3, +π/3]` for both joints (120° each) |
| **Buckets** | 9 arm × 13 hand = 117 discrete states |
| **Actions** | `'arm_up'`, `'arm_down'`, `'hand_up'`, `'hand_down'` (absent at range limits) |
| **Reward** | When the hand tip touches the ground (`hand_y >= ground_y - 10`): `reward = dx × 0.1` where `dx = old_hand_x - new_hand_x`. The body slides by `dx × 0.5`. If the hand is in the air: reward = 0. |
| **Terminal** | Never — `is_terminal()` always returns `False`. |
| **Initial state** | Midpoints `(4, 6)`, `robot_x = 200.0` |

**Forward kinematics:**
```
shoulder_x = robot_x + robot_width / 2
shoulder_y = ground_y - robot_height
elbow_x = shoulder_x + arm_length × cos(arm_angle)
elbow_y = shoulder_y - arm_length × sin(arm_angle)
hand_x  = elbow_x + hand_length × cos(arm_angle + hand_angle)
hand_y  = elbow_y - hand_length × sin(arm_angle + hand_angle)
```

`run_crawler(num_steps, epsilon, alpha, gamma, report_every)` is a
text-mode training driver that lazy-imports `QLearningAgent` from
`students.qlearning_agents`.

---

## Import Examples

```python
# Abstract interfaces
from env.mdp import MarkovDecisionProcess
from env.environment import Environment

# Grid world
from env.mars_grid import (
    MarsGrid,
    MarsGridEnvironment,
    build_mars_grid,
    load_grid_from_file,
    get_layout_path,
    GRID_REGISTRY,
    TERMINAL_STATE,
    NORTH, SOUTH, EAST, WEST, EXTRACT,
)

# Crawler
from env.crawler import CrawlerEnvironment, run_crawler
```

---

## Architecture Diagram

```
                          ┌──────────────┐
                          │   util.py    │
                          └──────┬───────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
       ┌──────▼──────┐                       ┌──────▼──────┐
       │   mdp.py    │                       │environment.py│
       └──────┬──────┘                       └──────┬──────┘
              │                                     │
       ┌──────▼─────────────────────────────────────▼──┐
       │              mars_grid.py                     │
       │  (MarsGrid, Environment, .lay loader,         │
       │   GRID_REGISTRY)                              │
       └──────┬────────────────────────────────────────┘
              │
    ┌─────────┼──────────┐
    │         │          │
    ▼         ▼          ▼
mars_rover  autograder  check
   .py        .py        .py

       ┌──────────────┐
       │environment.py │
       └──────┬────────┘
              │
       ┌──────▼──────┐
       │  crawler.py  │──────► display/crawler_gui.py
       └──────────────┘
```

Arrows indicate "imports from". All imports flow **downward** — no
circular dependencies exist within the `env/` package.
