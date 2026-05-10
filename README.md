[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
![Linux](https://img.shields.io/badge/Linux-compatible-FCC624?style=flat-square&logo=linux&logoColor=black)
![Windows](https://img.shields.io/badge/Windows-compatible-0078D6?style=flat-square&logo=windows&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-compatible-000000?style=flat-square&logo=apple&logoColor=white)

<div align="center">

# CS 451/551: Introduction to Artificial Intelligence

## Spring 2026

### Özyeğin University

</div>

---

---

<div align="center">

# 🚀🤖🔴 Homework 2: Reinforcement Learning — Student Guide

## Mars Rover: Red Planet Rescue

## Watch the Homework Description by Clicking on the Image

[![Watch the Homework Description](assets/cover3.png)](https://youtu.be/OdFavcX32mc?si=sO_y5GPRn-tJ_skq)

</div>

---

## 📑 Table of Contents

- [👋 Introduction](#-introduction)
- [🧩 Compatibility](#-compatibility)
- [🔑 Prerequisites](#-prerequisites)
- [🧬 Project Structure](#-project-structure)
- [📃 Files You Should Read (But Not Edit)](#-files-you-should-read-but-not-edit)
- [🗂️ Files You Will Edit](#️-files-you-will-edit)
  - [`students/value_iteration_agents.py` — Methods to Implement](#studentsvalue_iteration_agentspy--3-methods)
  - [`students/qlearning_agents.py` — Methods to Implement](#studentsqlearning_agentspy--7-methods)
  - [`students/analysis.py` — Functions to Implement](#studentsanalysispy--5-functions)
- [🔧 Installation](#-installation)
- [🚀 Getting Started](#-getting-started)
  - [Play the Grid World Manually](#step-1-play-the-grid-world-manually)
  - [Try Different Layouts](#step-2-try-different-layouts)
  - [Understand the Mars Grid MDP](#step-3-understand-the-mars-grid-mdp)
  - [Understand Movement Noise](#step-4-understand-movement-noise-solar-interference)
  - [Understand Key Parameters](#step-5-understand-the-key-parameters)
- [📝 Your Tasks (5 Questions — 75 Points Total)](#-your-tasks-5-questions--75-points-total)
  - [Q1: Value Iteration](#question-1-value-iteration-20-pts)
  - [Q2: Mission Profiles — Policy Analysis](#question-2-mission-profiles--policy-analysis-20-pts)
  - [Q3: Q-Learning](#question-3-q-learning-20-pts)
  - [Q4: Epsilon-Greedy Exploration](#question-4-epsilon-greedy-exploration-8-pts)
  - [Q5: Q-Learning Deployment on Mars](#question-5-q-learning-deployment-on-mars-7-pts)
- [🧐 Testing Your Implementation](#-testing-your-implementation)
- [📟 Useful Command-Line Options](#-useful-command-line-options)
- [💡 General Algorithm Hints](#-general-algorithm-hints)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [💯 Points Breakdown](#-points-breakdown)
- [Academic Integrity](#academic-integrity)
- [🤝 Attribution](#-attribution)
- [🔗 Useful Links](#-useful-links)

---

## 👋 Introduction

In this project, your Mars Rover agent will navigate treacherous Martian terrain — avoiding deep craters and collecting
scattered geological samples. You will implement **Value Iteration** and **Q-Learning** (with epsilon-greedy
exploration) to teach Rover-7 to act optimally under stochastic solar interference.

The grid world engine, display system, and test infrastructure are already built for you. **Your job is to implement the
reinforcement learning algorithms that make Rover-7 smart.**

### 📖 The Story

The year is 2089. **Rover-7**, the Mars Exploration Agency's most advanced robot, has just landed in the Valles
Marineris region. A massive Martian dust storm has scattered invaluable geological samples across kilometers of dangerous
terrain. Some samples landed near deep craters. Others ended up on the far side of rocky walls.

To make matters worse, **solar interference** from the Martian atmosphere disrupts Rover-7's electric motors. When the
rover tries to move north, there is a **20% chance** it will slip sideways — 10% east, 10% west. This stochastic noise
is what makes simple planning insufficient and reinforcement learning necessary.

You will solve this problem in three phases:

1. **Phase 1 — Offline Planning (Q1–Q2):** With satellite terrain maps, compute optimal policies using Value Iteration
   before the rover touches the ground.
2. **Phase 2 — Learning by Experience (Q3–Q4):** The satellite link drops. Rover-7 must learn to navigate by trial and
   error using tabular Q-Learning with epsilon-greedy exploration.
3. **Phase 3 — Deployment (Q5):** Run the same Q-Learner on Mars grids and the Crawler robot, verifying it generalises
   beyond the grid world it was tuned on.

---

## 🧩 Compatibility

This project has been tested and works on **Linux, macOS, and Windows**.

---

## 🔑 Prerequisites

- **Python 3.10+** (no external packages required — only the Python standard library)
- **Pygame** (optional — for the animated Mars GUI display)

> **Note:** The autograder, self-check, and all tests run **without** Pygame. The GUI is purely for visualization and
> exploration. Everything works in text mode automatically if Pygame is not installed.

To verify your setup, run:

```bash
python mars_rover.py -m
```

If Pygame is installed, you'll see an animated Mars surface with Rover-7. Use the arrow keys to control the rover
manually. Without Pygame, text mode activates automatically.

```bash
python mars_rover.py -t   # Force text mode (no Pygame needed)
```

If you want the animated GUI:

```bash
pip install pygame
```

---

## 🧬 Project Structure

```
mars_rover_rl/
├── mars_rover.py                  # Entry point — run this to start
├── autograder.py                   # Official autograder (75 pts)
├── check.py                       # Self-check (3 maps, no grade)
├── project_params.py               # Project metadata
├── util.py                         # Data structures: Counter, sampling
│
├── env/                            # Environment package
│   ├── mdp.py                     # Abstract MarkovDecisionProcess interface
│   ├── environment.py             # Abstract Environment interface
│   ├── mars_grid.py               # MarsGrid MDP, layout loader, grid registry
│   └── crawler.py                 # Crawler robot domain (second RL domain)
│
├── agents/                         # Agent base classes (do NOT edit)
│   └── learning_agents.py         # ValueEstimationAgent, ReinforcementAgent
│
├── students/                       # ✏ YOUR CODE — only edit these 3 files
│   ├── value_iteration_agents.py  # TODO — Value Iteration agent (Q1)
│   ├── qlearning_agents.py        # TODO — Q-Learning agent (Q3, Q4, Q5)
│   └── analysis.py                # TODO — Policy parameter choices (Q2)
│
├── display/                        # Visualization (optional — requires Pygame)
│   ├── gui_display.py             # Pygame animated Mars display
│   ├── text_display.py            # ASCII terminal display
│   └── crawler_gui.py             # Pygame crawler robot display
│
├── grading/                        # Test framework
│   ├── grading.py                 # Grade tracker
│   ├── test_parser.py             # .test/.solution file parser
│   ├── test_classes.py            # Question/TestCase base classes
│   └── mars_test_classes.py       # Project-specific test logic
│
├── layouts/                        # 9 grid layout files (.lay)
│   ├── tiny_grid.lay              # 1×3 — minimal sanity check
│   ├── small_grid.lay             # 3×2 — basic Q-learning
│   ├── base_camp_grid.lay         # 4×3 — classic grid (like textbook)
│   ├── canyon_grid.lay            # 5×4 — policy analysis (Q2)
│   ├── bridge_grid.lay             # 7×3 — narrow risky passage
│   ├── maze_grid.lay              # 4×5 — long corridors
│   ├── cliff_grid.lay             # 5×4 — cliff walking
│   ├── medium_grid.lay            # 5×5 — multiple hazards
│   └── expedition_grid.lay        # 7×6 — multi-objective
│
└── test_cases/                     # Autograder test data (q1–q5)
    └── q1/ ... q5/                 # CONFIG + .test + .solution per question
```

---

## 📃 Files You Should Read (But Not Edit)

<div align="center">

| File                          | Description                                                                                              |
|-------------------------------|----------------------------------------------------------------------------------------------------------|
| `util.py`                     | `Counter` class (dict with default 0), `flip_coin(p)`, `sample()` — you'll use these in Q-Learning      |
| `env/mdp.py`                  | Abstract `MarkovDecisionProcess` — defines `get_states()`, `get_transition_states_and_probs()`, etc.     |
| `env/mars_grid.py`            | `MarsGrid` MDP implementation — the grid world your agents operate in                                    |
| `agents/learning_agents.py`   | `ValueEstimationAgent` and `ReinforcementAgent` — base classes your agents extend                        |

</div>

---

## 🗂️ Files You Will Edit

You only need to modify **three files** inside the `students/` directory. All places where you need to write code are
marked with `# *** YOUR CODE HERE ***` comments.

<div align="center">

| File                                   | What You Implement                                         |
|----------------------------------------|------------------------------------------------------------|
| `students/value_iteration_agents.py`   | Value Iteration agent (Q1)                                 |
| `students/qlearning_agents.py`         | Q-Learning agent + ε-greedy exploration (Q3, Q4, Q5)       |
| `students/analysis.py`                 | Policy parameter choices for the Canyon Grid (Q2)          |

</div>

⚠️🚨 **Do not modify any other files.**

### Exact methods and functions to implement

#### `students/value_iteration_agents.py` — 3 methods

<div align="center">

| Method                                       | Question | Description                                                    |
|----------------------------------------------|----------|----------------------------------------------------------------|
| `run_value_iteration()`                      | Q1       | Main VI loop — batch Bellman updates for *k* iterations        |
| `compute_q_value_from_values(state, action)` | Q1       | Q(s,a) = Σ T(s,a,s') · [R(s,a,s') + γ · V(s')]               |
| `compute_action_from_values(state)`          | Q1       | Best action from current values (argmax over Q-values)         |

</div>

#### `students/qlearning_agents.py` — 5 methods

<div align="center">

| Class / Method                                                | Question | Description                                                         |
|---------------------------------------------------------------|----------|---------------------------------------------------------------------|
| `QLearningAgent.get_q_value(state, action)`                   | Q3       | Return Q-value from Counter (default 0 for unseen pairs)            |
| `QLearningAgent.compute_value_from_q_values(state)`           | Q3       | V(s) = max_a Q(s,a); return 0.0 if terminal                        |
| `QLearningAgent.compute_action_from_q_values(state)`          | Q3       | Best action — **break ties randomly** with `random.choice()`        |
| `QLearningAgent.get_action(state)`                            | Q4       | ε-greedy: random with prob ε, else best action                      |
| `QLearningAgent.update(state, action, next_state, reward)`    | Q3       | Q-learning update: Q(s,a) ← Q(s,a) + α·[r + γ·V(s') − Q(s,a)]    |

</div>

#### `students/analysis.py` — 5 functions

<div align="center">

| Function       | Question | Description                                                       |
|----------------|----------|-------------------------------------------------------------------|
| `question2a()` | Q2       | Return `(discount, noise, living_reward)` → close exit, risk cliff|
| `question2b()` | Q2       | Close exit, avoid cliff                                           |
| `question2c()` | Q2       | Distant depot (+10), risk cliff                                   |
| `question2d()` | Q2       | Distant depot (+10), avoid cliff                                  |
| `question2e()` | Q2       | Never terminate (wander forever)                                  |

</div>

---

## 🔧 Installation

### Prerequisites

- Python 3.10 or later
- pip

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/A-D-Alamdari/CS451-551-Homework2.git
cd CS451-551-Homework2

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install Pygame (optional — for animated GUI)
pip install pygame
```

---

## 🚀 Getting Started

### Step 1: Play the Grid World Manually

```bash
python mars_rover.py -m
```

Use the arrow keys to move Rover-7 across the Martian grid. Press Enter/Space to extract at a sample site or crater.
This helps you understand the grid mechanics — especially how **solar interference** makes you slip sideways 20% of the
time.

### Step 2: Try Different Layouts

```bash
python mars_rover.py -m -g base_camp_grid
python mars_rover.py -m -g canyon_grid
python mars_rover.py -m -g maze_grid
python mars_rover.py --list-layouts        # See all 9 available grids
```

### Step 3: Understand the Mars Grid MDP

Every grid cell is one of:

- **Open terrain** (`_`) — the rover moves through freely
- **Rock wall** (`#`) — impassable; rover bounces back and stays in place
- **Sample site** (positive number: `+1`, `+5`, `+10`, `+25`) — terminal state with positive reward
- **Crater** (negative number: `-1`, `-5`, `-10`, `-100`) — terminal state with negative reward
- **Start** (`S`) — where Rover-7 begins each episode

When the rover reaches a terminal cell (sample or crater), the only action is `extract`, which collects the sample (or
falls into the crater) and ends the episode.

### Step 4: Understand Movement Noise (Solar Interference)

When Rover-7 chooses to go **north**:

- **80%** chance it actually goes north (intended direction)
- **10%** chance it slips **west** (perpendicular)
- **10%** chance it slips **east** (perpendicular)

If any movement would push the rover into a wall or off the grid, the rover stays in place. This means standing next to
a crater is dangerous even if you never intentionally move toward it — noise might push you in.

### Step 5: Understand the Key Parameters

```bash
python mars_rover.py -a value --discount 0.9 --noise 0.2 --living-reward 0.0
```

- **Discount (γ)** — how much future rewards matter (0 = myopic, 1 = far-sighted)
- **Noise** — probability of slipping sideways (0 = deterministic, 0.2 = default)
- **Living reward** — reward per non-terminal step (negative = hurry, positive = wander forever)

---

## 📝 Your Tasks (5 Questions — 75 Points Total)

### Question 1: Value Iteration (20 pts)

**File:** `students/value_iteration_agents.py` — class `ValueIterationAgent`

Implement the Value Iteration algorithm — an offline planner that computes optimal values using the Bellman equation:

```
V_{k+1}(s) = max_a  Σ_{s'} T(s,a,s') · [R(s,a,s') + γ · V_k(s')]
```

**You implement three methods:**

1. `run_value_iteration()` — the main loop. For each iteration, create a **new** `Counter` for V_{k+1}, compute values
   for all states from V_k, then replace.
2. `compute_q_value_from_values(state, action)` — compute Q(s,a) from the current values.
3. `compute_action_from_values(state)` — return the best action (argmax over Q-values). Return `None` for terminal
   states.

**Key requirements:**

- Use **batch** updates — create a new `Counter()` each iteration and compute ALL new values from old values before
  replacing `self.values`. In-place updates give wrong results.
- Handle terminal states (no actions available → value stays 0).

**Try it out:**

```bash
python mars_rover.py -a value -i 100 -k 10           # 100 VI sweeps, 10 test episodes
python mars_rover.py -a value -i 5 -g base_camp_grid  # Watch 5 iterations converge
python autograder.py -q q1                              # Grade Q1
```

**Hint:** After running Value Iteration, press **V** in the GUI to see values, **Q** for Q-values, and **P** for policy
arrows.

---

### Question 2: Mission Profiles — Policy Analysis (20 pts)

**File:** `students/analysis.py` — functions `question2a()` through `question2e()`

*Depends on: Question 1 (Value Iteration)*

The **Canyon Grid** has a close exit (+1), a distant sample depot (+10), and a crater cliff (-10 each along the bottom):

```
  [ ] [ ] [ ] [ ] [+1]     ← close exit (small reward)
  [##] [ ] [ ] [ ] [+10]    ← distant depot (big reward)
  [S ] [ ] [ ] [ ] [ ]     ← start
  [##] [-10][-10][-10][-10]  ← crater cliff
```

Choose `(discount, noise, living_reward)` to produce each behavior:

<div align="center">

| Function       | Desired Behavior                                    |
|----------------|-----------------------------------------------------|
| `question2a()` | Prefer close exit (+1), **risking** the crater cliff|
| `question2b()` | Prefer close exit (+1), **avoiding** the cliff      |
| `question2c()` | Prefer distant depot (+10), **risking** the cliff   |
| `question2d()` | Prefer distant depot (+10), **avoiding** the cliff  |
| `question2e()` | Avoid all exits and craters (never terminate)       |

</div>

Each function returns a `(discount, noise, living_reward)` tuple. If a behavior is impossible, return
`'NOT POSSIBLE'`.

**Try it out:**

```bash
python mars_rover.py -g canyon_grid -a value --discount 0.9 --noise 0.2 --living-reward 0.0
python autograder.py -q q2
```

**Hint:** Think about what makes the rover prefer near vs. far rewards (discount), what makes it avoid danger (noise),
and what makes it never want to stop (positive living reward). Experiment in the GUI.

---

### Question 3: Q-Learning (20 pts)

**File:** `students/qlearning_agents.py` — class `QLearningAgent`

Implement a model-free reinforcement learning agent. Unlike Value Iteration, Q-Learning has **no model** of the
environment — the rover learns entirely from trial and error.

The Q-learning update rule:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_{a'} Q(s',a') − Q(s,a)]
```

**You implement five methods:**

1. `get_q_value(state, action)` — lookup Q-value from Counter (returns 0 for unseen pairs)
2. `compute_value_from_q_values(state)` — max Q over legal actions; 0.0 if terminal
3. `compute_action_from_q_values(state)` — best action; **break ties randomly!**
4. `update(state, action, next_state, reward)` — the Q-learning update
5. `get_action(state)` — see Q4 below (implement both Q3 and Q4 in the same file)

**Key requirements:**

- **Break ties randomly** in `compute_action_from_q_values` using `random.choice()`.
- Actions the rover has **never tried** have Q-value = 0 (Counter default). If all tried actions have negative Q-values,
  an **untried action (Q=0) is the best choice!** Don't skip unseen actions.
- Only access Q-values through `self.get_q_value()`, never `self.q_values[...]` directly — this keeps the accessor
  contract as the single public way to read a Q-value.

**Try it out:**

```bash
python mars_rover.py -a q -k 5 -m         # Manual Q-learning, 5 episodes
python mars_rover.py -a q -k 100           # Train 100 episodes
python autograder.py -q q3
```

**Hint:** Q-Learning "leaves learning in its wake" — the rover updates the state it just **left**, not the one it moved
to. Watch this happen in the GUI.

---

### Question 4: Epsilon-Greedy Exploration (8 pts)

**File:** `students/qlearning_agents.py` — method `QLearningAgent.get_action(state)`

*Depends on: Question 3*

Implement the ε-greedy exploration strategy:

- With probability **ε**: take a **random** legal action (explore)
- With probability **1 − ε**: take the **best** action from Q-values (exploit)

**Key requirements:**

- Use `util.flip_coin(self.epsilon)` to decide explore vs. exploit
- Use `random.choice(legal_actions)` for random selection
- A random action **may** happen to be the best action — that's fine. Don't filter out the optimal action during
  exploration.
- Return `None` if no legal actions exist

**Try it out:**

```bash
python mars_rover.py -a q -k 100                            # Watch Q-learning with ε=0.3
python mars_rover.py -a q -k 100 --noise 0.0 -e 0.1         # Low exploration
python mars_rover.py -a q -k 100 --noise 0.0 -e 0.9         # High exploration
python autograder.py -q q4
```

After implementing Q4, the **Mars Crawler** robot should also work without any code changes:

```bash
python -m env.crawler                # Crawler learns to crawl forward
```

If this doesn't work, your Q-learning is probably too specific to the grid world. Make it generic.

---

### Question 5: Q-Learning Deployment on Mars (7 pts)

**File:** No new code needed — tests your Q3 + Q4 implementation

*Depends on: Questions 3 + 4*

Deploy your Q-learner on the `small_grid`. Train for 2000 episodes, then test for 100. Must win ≥ 80%.

```bash
python mars_rover.py -a q -k 2010 -g small_grid -e 0.05 --alpha 0.2 --discount 0.8 -q
python autograder.py -q q5
```

The autograder also verifies your Q-learning works on the **Crawler** robot domain — a two-armed robot that learns
crawling locomotion by adjusting joint angles.

> **Why larger grids fail:** On `medium_grid` or larger, tabular Q-learning struggles to converge — every cell is a
> separate state with separate Q-values. The rover has no way to generalise that "craters are bad everywhere." This
> motivates function approximation (out of scope for this homework).

---

## 🧐 Testing Your Implementation

### Running the Self-Check (No Grade)

The file `check.py` tests your implementation on **3 maps** (small, base camp, canyon) and shows diagnostic pass/fail
output. **No points or grades** — purely for debugging.

```bash
python check.py                 # Run all checks
python check.py -q q1           # Check only Q1
```

### Running the Official Autograder (75 Points)

```bash
python autograder.py                     # Grade all questions
python autograder.py -q q1               # Grade one question
python autograder.py --no-graphics       # Skip any display
```

The autograder tests across **all 9 grid layouts** with pre-computed reference solutions. Values must match within 0.01
tolerance.

### What the Autograder Checks

<div align="center">

| Question | What It Verifies                                                                          |
|----------|-------------------------------------------------------------------------------------------|
| Q1       | V(s), Q(s,a), and π(s) at iterations 0, 1, 2, and 100 on multiple grids                  |
| Q2       | Policy at key cells matches expected direction for each parameter tuple                    |
| Q3       | Q-values after fixed training episodes match reference (with seeded randomness)            |
| Q4       | Best-action frequency falls in expected range for given ε; pure exploit at ε=0             |
| Q5       | Win rate ≥ 80% on small_grid; crawler robot moves forward                                 |

</div>

### Suggested Workflow

1. **Start with Q1** — implement Value Iteration and run:
   ```bash
   python check.py -q q1
   python autograder.py -q q1
   ```
2. **Move to Q2** — experiment with parameters in the GUI, then fill in `analysis.py`.
3. **Q3 + Q4 together** — implement Q-Learning and ε-greedy (they share the same file).
4. **Q5 is free** — if Q3 + Q4 pass, Q5 usually passes automatically.

After each question, run the tests to confirm everything passes before moving on.

---

## 📟 Useful Command-Line Options

```bash
python mars_rover.py --help              # Show all options

# Layout selection
python mars_rover.py -g canyon_grid      # Use a specific grid
python mars_rover.py -g expedition_grid  # Larger multi-objective grid
python mars_rover.py --list-layouts      # List all 9 available grids

# Agent selection
python mars_rover.py -a value -i 100     # Value Iteration, 100 sweeps
python mars_rover.py -a q -k 500         # Q-Learning, 500 episodes
python mars_rover.py -a random            # Random agent

# MDP parameters
python mars_rover.py --discount 0.9 --noise 0.2 --living-reward -0.04
python mars_rover.py -e 0.3 --alpha 0.5   # Q-learning: ε and α

# Display options
python mars_rover.py -m                   # Manual control (arrow keys)
python mars_rover.py -t                   # Text mode (no Pygame)
python mars_rover.py -q                   # Quiet (no transition output)

# GUI keyboard shortcuts (when using Pygame display):
# V = show values   Q = show Q-values   P = show policy   ESC = quit
```

---

## 💡 General Algorithm Hints

### Value Iteration vs. Q-Learning

<div align="center">

| Property     | Value Iteration (Q1)            | Q-Learning (Q3)                 |
|--------------|---------------------------------|---------------------------------|
| Has model?   | Yes — knows T(s,a,s') and R     | No — learns from experience     |
| Planning     | Offline (before acting)         | Online (while acting)           |
| Updates      | Sweeps over ALL states          | Updates ONE (s,a) per step      |
| Convergence  | After enough iterations         | After enough episodes           |

</div>

### The Q-Learning Update Pattern

```
1. Observe current state s
2. Choose action a (ε-greedy)
3. Execute action, observe next state s' and reward r
4. Update: Q(s,a) ← Q(s,a) + α · [r + γ · max_{a'} Q(s',a') − Q(s,a)]
5. s ← s'
6. Repeat until episode ends
```

### Common Mistakes to Avoid

- **In-place VI updates (Q1):** Create a **new** `Counter()` per iteration. Reading and writing `self.values` in the
  same sweep gives wrong results.
- **Not breaking ties randomly (Q3):** If two actions have the same Q, use `random.choice()`. Without this, the rover
  gets stuck in deterministic loops.
- **Forgetting unseen actions have Q=0 (Q3):** The `Counter` returns 0 for missing keys. If all tried actions have
  Q < 0, an untried action (Q=0) is optimal.
- **Accessing Q-values directly:** Always call `self.get_q_value()`, never `self.q_values[...]`. The accessor is the
  single public read path; future subclasses may change the underlying representation.

---

## 🛠️ Troubleshooting

### "Method not implemented" / `NotImplementedError`

This means you haven't filled in the `# *** YOUR CODE HERE ***` section yet. The stub calls
`util.raise_not_defined()`, which raises `NotImplementedError`. Replace it with your implementation.

### V(start) is 0 after Value Iteration

You're using in-place updates. Create a **new** `Counter()` for V_{k+1} each iteration, then replace `self.values` at
the end of each sweep.

### Q-Learning rover gets stuck

Check two things:

1. Are you **breaking ties randomly** in `compute_action_from_q_values`?
2. Are **untried actions (Q=0)** being considered? If all tried actions have negative Q, an untried action is best.

### Q5 fails but Q3/Q4 pass

Q5 uses different learning parameters (ε=0.05, α=0.2, γ=0.8). Your Q-learning should work with any parameter values,
not just the defaults.

### Crawler doesn't move

Your Q-learning agent is too specific to the grid world. Make sure `get_action`, `update`, etc. don't assume anything
about the state format — it's `(arm_index, hand_index)` for the crawler, not `(x, y)`.

### Tests pass for Q1–Q4 but Q2 is wrong

Q2 depends on Q1. Make sure your Value Iteration is fully correct before tuning Q2 parameters. Run:

```bash
python autograder.py -q q1
```

### `TypeError: unhashable type: 'list'`

If you see this in any state representation, convert lists to tuples. Python sets and dict keys require hashable types.

### No graphics / Pygame errors

```bash
python mars_rover.py -t              # Text-only mode (always works)
python mars_rover.py -q              # Quiet mode (no display at all)
```

The autograder and `check.py` never require Pygame — they always work headless.

---

## 💯 Points Breakdown

<div align="center">

| Question          | Topic                                  | Points  |
|-------------------|----------------------------------------|---------|
| Q1                | Value Iteration                        | 20      |
| Q2                | Policy Analysis (Canyon Grid)          | 20      |
| Q3                | Q-Learning                             | 20      |
| Q4                | Epsilon-Greedy Exploration             | 8       |
| Q5                | Q-Learning Deployment + Crawler        | 7       |
| **Code Subtotal** |                                        | **75**  |
| Report            | Written report                         | 25      |
| **Total**         |                                        | **100** |

</div>

---

## Academic Integrity

- Do **not** distribute or publish solutions
- Do **not** copy solutions from the internet
- You **may** discuss general approaches with classmates, but write your own code

---

## 🤝 Attribution

This project is an **original work** created for CS 451/551 at Özyeğin University. It is inspired by the reinforcement
learning concepts covered in the UC Berkeley CS 188 curriculum but uses an entirely original codebase, theme, and
project structure.

- **Mars Rover theme, grid world engine, autograder, and all code** — created by Amin D. Alamdari
  (amin.alamdari@ozu.edu.tr), 2026
- **Reinforcement learning algorithms** — based on standard textbook formulations from Sutton & Barto,
  *Reinforcement Learning: An Introduction*
- **Pedagogical structure** — inspired by the UC Berkeley CS 188 Pac-Man AI Projects by John DeNero and Dan Klein

---

## 🔗 Useful Links

- [Homework's Github Repository](https://github.com/A-D-Alamdari/CS451-551-Homework2)
- [Homework Page](https://a-d-alamdari.github.io/mars-rover-ai-hw2.html)
- [Sutton & Barto — Reinforcement Learning: An Introduction (free PDF)](http://incompleteideas.net/book/the-book-2nd.html)
  - [Chapter 4: Dynamic Programming (Value Iteration)](http://incompleteideas.net/book/ebook/node43.html)
  - [Chapter 6: Temporal-Difference Learning (Q-Learning)](http://incompleteideas.net/book/ebook/node65.html)
- [UC Berkeley CS 188 — Reinforcement Learning Lectures](https://inst.eecs.berkeley.edu/~cs188/fa24/)
- [David Silver — RL Course Lecture 4: Model-Free Prediction](https://youtu.be/PnHCvfgC_ZA)
- [David Silver — RL Course Lecture 5: Model-Free Control](https://youtu.be/0g4j2k_Ggc4)
