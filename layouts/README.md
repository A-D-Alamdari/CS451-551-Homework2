# layouts/ — Grid Layout Files

## Format

Each `.lay` file defines a Mars terrain grid:

- One row per line, cells separated by spaces
- `_` = open terrain (the rover can pass through)
- `#` = rock wall (impassable — rover stays in place on collision)
- `S` = start position (also open terrain; the rover's landing site)
- Positive numbers = sample sites (geological samples, mission reward)
- Negative numbers = craters (mission-ending hazards)
- `//` = comment line (ignored by the parser)
- Blank lines are ignored
- All rows must be the same width

If no `S` cell exists, the start defaults to `(0, 0)` (bottom-left
after the y-flip).

---

## Grid Catalog

### tiny_grid.lay (1×3, 3 cells) — Beginner

The simplest non-trivial MDP: one column, three rows. Only two
actions are ever legal (north or south), and the agent must learn
to go toward the +10 sample and away from the -10 crater. Ideal
for sanity-checking VI and Q-learning before scaling up.

```
-10       ← crater (y=2, top)
 S        ← start  (y=1, middle)
 10       ← sample (y=0, bottom)
```

| Detail | Value |
|---|---|
| Dimensions | 1 × 3 |
| Walls | 0 |
| Samples | 1 (+10) |
| Craters | 1 (-10) |
| Used in | Q1, Q3 autograder tests |
| Purpose | Minimal sanity check |

---

### small_grid.lay (3×2, 6 cells) — Beginner

Minimal 2-D navigation. A wall blocks the direct diagonal path,
forcing the rover to choose between going north first (toward the
+1 sample) or east first (risking the -1 crater). The workhorse
test grid for Q-learning convergence checks.

```
#  _  1       ← wall, open, sample (y=1)
S  _ -1       ← start, open, crater (y=0)
```

| Detail | Value |
|---|---|
| Dimensions | 3 × 2 |
| Walls | 1 |
| Samples | 1 (+1) |
| Craters | 1 (-1) |
| Used in | Q1, Q3, Q5 autograder tests; check.py |
| Purpose | Minimal 2-D grid for Q-learning convergence tests |

---

### base_camp_grid.lay (4×3, 12 cells) — Beginner

The classic "BookGrid" analogue. A single wall in the centre
creates two corridors. The +1 sample is at the bottom-right and
the -1 crater at the top-right; the rover starts at the
bottom-left. No `S` marker — start defaults to `(0, 0)`.

```
 _  _  _ -1       ← crater at (3,2)
 _  #  _  _       ← wall at (1,1)
 _  _  _  1       ← sample at (3,0), start at (0,0)
```

| Detail | Value |
|---|---|
| Dimensions | 4 × 3 |
| Walls | 1 |
| Samples | 1 (+1) |
| Craters | 1 (-1) |
| Used in | Q1, Q3 autograder tests; check.py |
| Purpose | Standard grid for value iteration demonstrations |

---

### canyon_grid.lay (5×4, 20 cells) — Intermediate

The discounting / noise / living-reward analysis grid used by Q2.
A close +1 exit sits at the top-right, a much more valuable +10
depot at the middle-right, and a crater cliff of -10 tiles runs
along the entire bottom row. The rover must choose: short safe
path to +1, risky cliff run to +10, safe upper detour to +10, or
wander forever.

```
 _  _  _  _  1       ← close exit (y=3)
 #  _  _  _ 10       ← wall + distant depot (y=2)
 S  _  _  _  _       ← start at (0,1) (y=1)
 # -10 -10 -10 -10   ← crater cliff (y=0)
```

| Detail | Value |
|---|---|
| Dimensions | 5 × 4 |
| Walls | 2 |
| Samples | 2 (+1, +10) |
| Craters | 4 (-10 each) |
| Used in | Q1, Q2 (all 5 profiles), Q3 autograder tests; check.py |
| Purpose | Policy analysis — 5 distinct behaviours from tuning 3 knobs |

---

### bridge_grid.lay (7×3, 21 cells) — Intermediate

A single-row corridor flanked above and below by -100 craters.
The rover starts one step east of a safe +1 exit and must decide:
grab the +1 immediately (safe, low reward) or walk 5 steps east
across the bridge to the +10 exit (risky — any slip off the bridge
is catastrophic).

```
 # -100 -100 -100 -100 -100  #       ← crater walls (y=2)
 1   S    _    _    _    _  10       ← corridor with exits (y=1)
 # -100 -100 -100 -100 -100  #       ← crater walls (y=0)
```

| Detail | Value |
|---|---|
| Dimensions | 7 × 3 |
| Walls | 4 |
| Samples | 2 (+1, +10) |
| Craters | 10 (-100 each) |
| Used in | Q3 autograder tests |
| Purpose | Risk-reward trade-off under extreme penalties |

---

### maze_grid.lay (4×5, 20 cells) — Intermediate

A thin vertical wall down column 1 forces the rover to navigate
around it. Start at the bottom-left, +1 sample at the top-right.
No craters — the only challenge is finding the path.

```
 _  _  _  1       ← sample at (3,4)
 _  #  _  _       ← wall column
 _  #  _  _
 _  #  _  _
 S  _  _  _       ← start at (0,0)
```

| Detail | Value |
|---|---|
| Dimensions | 4 × 5 |
| Walls | 3 |
| Samples | 1 (+1) |
| Craters | 0 |
| Used in | GRID_REGISTRY (available via `mars_rover.py -g maze`) |
| Purpose | Path-finding around obstacles |

---

### cliff_grid.lay (5×4, 20 cells) — Intermediate

The classic cliff-walking problem. The rover starts at the
left-middle, the +1 sample is at the right-middle, and the entire
bottom row is a crater cliff. The shortest path hugs the cliff
(risky with noise); the safe path goes along the top.

```
 _  _  _  _  _       ← safe top row (y=3)
 _  _  _  _  _       ← open (y=2)
 S  _  _  _  1       ← start + sample (y=1)
-10 -10 -10 -10 -10  ← crater cliff (y=0)
```

| Detail | Value |
|---|---|
| Dimensions | 5 × 4 |
| Walls | 0 |
| Samples | 1 (+1) |
| Craters | 5 (-10 each) |
| Used in | GRID_REGISTRY (available via `mars_rover.py -g cliff_grid`) |
| Purpose | Cliff-walking risk/safety trade-off |

---

### medium_grid.lay (5×5, 25 cells) — Intermediate

Mid-sized world with walls forming corridors, one +1 sample at the
top-right, and two -1 craters. The start is at the bottom-left
region. Tests generalisation: the agent must learn to navigate
corridors and avoid distributed hazards.

```
 _  _  _  _  1       ← sample at (4,4)
 _  #  #  _  _       ← wall corridor
 _  _  _  _ -1       ← crater at (4,2)
 S  _  #  _  _       ← start at (0,1), wall
 _  _  _  _ -1       ← crater at (4,0)
```

| Detail | Value |
|---|---|
| Dimensions | 5 × 5 |
| Walls | 3 |
| Samples | 1 (+1) |
| Craters | 2 (-1 each) |
| Used in | GRID_REGISTRY (available via `mars_rover.py -g medium_grid`) |
| Purpose | Corridor navigation with distributed hazards |

---

### expedition_grid.lay (7×6, 42 cells) — Advanced

The largest grid. Multiple sample sites of varying value (+5, +3,
+2), a single -5 crater, and walls partitioning the map into
corridors. Tests value iteration scalability across a complex
multi-objective map.

```
 _  _  _  _  _  _  5       ← high-value sample (y=5)
 _  #  #  #  #  _  _       ← wall barrier
 _  _  _  2  _  _  _       ← mid-value sample (y=3)
 _  #  _  #  #  #  _       ← wall corridor
 _  _  _  _  _ -5  _       ← crater (y=1)
 S  _  #  _  _  _  3       ← start + sample (y=0)
```

| Detail | Value |
|---|---|
| Dimensions | 7 × 6 |
| Walls | 9 |
| Samples | 3 (+5, +3, +2) |
| Craters | 1 (-5) |
| Used in | GRID_REGISTRY (available via `mars_rover.py -g expedition_grid`) |
| Purpose | Multi-objective stress test for value iteration |

---

## Summary Table

| Grid | Size | Cells | Walls | +Terms | -Terms | Start | Difficulty |
|---|---|---|---|---|---|---|---|
| tiny_grid | 1×3 | 3 | 0 | 1 | 1 | `(0,1)` | Beginner |
| small_grid | 3×2 | 6 | 1 | 1 | 1 | `(0,0)` | Beginner |
| base_camp_grid | 4×3 | 12 | 1 | 1 | 1 | `(0,0)` | Beginner |
| canyon_grid | 5×4 | 20 | 2 | 2 | 4 | `(0,1)` | Intermediate |
| bridge_grid | 7×3 | 21 | 4 | 2 | 10 | `(1,1)` | Intermediate |
| maze_grid | 4×5 | 20 | 3 | 1 | 0 | `(0,0)` | Intermediate |
| cliff_grid | 5×4 | 20 | 0 | 1 | 5 | `(0,1)` | Intermediate |
| medium_grid | 5×5 | 25 | 3 | 1 | 2 | `(0,1)` | Intermediate |
| expedition_grid | 7×6 | 42 | 9 | 3 | 1 | `(0,0)` | Advanced |

---

## Adding New Layouts

1. **Create the file:**
   ```
   // my_grid.lay -- description
   _ _ _ 1
   _ # _ _
   S _ _ -1
   ```

2. **Verify it parses:**
   ```bash
   python -c "from env.mars_grid import load_grid_from_file; print(load_grid_from_file('layouts/my_grid.lay'))"
   ```

3. **Use it immediately:**
   ```bash
   python mars_rover.py -g my_grid
   ```
   `build_mars_grid` automatically finds `.lay` files in `layouts/`.

4. **Optionally add to the autograder:** create a `.test` file in
   `test_cases/qN/` and a matching `.solution` file populated from a
   reference implementation, then verify with `python autograder.py -q qN`.
