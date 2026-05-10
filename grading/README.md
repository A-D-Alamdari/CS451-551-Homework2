# grading/ — Test Framework & Autograder Engine

## Purpose

A data-driven testing framework that reads `.test` and `.solution`
files from `test_cases/`, instantiates project-specific test-case
classes, runs the student's code, compares outputs against reference
values, and accumulates a 75-point grade across 5 questions.
Students never need to look at this package — they interact with
it via `python autograder.py`.

---

## Architecture

```
autograder.py (root)
│
│  reads test_cases/qN/
│  ┌──────────────────────────────────────────────────┐
│  │ CONFIG           → question type + max points    │
│  │ vi_tiny.test     → test parameters               │
│  │ vi_tiny.solution → expected reference values      │
│  └──────────────────────────────────────────────────┘
│
│  uses grading/
│  ┌──────────────────────────────────────────────────┐
│  │ test_parser.py      parse .test / .solution      │
│  │ test_classes.py     Question + TestCase bases     │
│  │ mars_test_classes.py  project-specific logic      │
│  │ grading.py          Grades tracking + output      │
│  └──────────────────────────────────────────────────┘
│
│  imports student code from students/
│  compares student output against solution data
│
▼  prints results + total/75
```

**Data flow for a single test:**

1. `autograder.py` reads `CONFIG` to determine the question class
   (`NumberPassedQuestion`) and max points.
2. For each `.test` file, it parses the key-value pairs with
   `test_parser.parse_test` and looks up the test-class name
   (e.g. `"ValueIterationTest"`) in `TEST_CLASS_MAP`.
3. The matching `.solution` file (if non-empty) is parsed
   similarly.
4. The test-class `execute()` method builds the grid, constructs
   the student agent, runs it, and compares its output against
   the solution data.
5. The result (pass/fail + message) is recorded in the `Grades`
   object.
6. After all tests, `Grades.produce_output()` prints the table.

---

## Files

### test_parser.py (156 lines)

Parses Berkeley-style key-value data files into plain `dict`s.

**Functions:**

| Function | Description |
|---|---|
| `parse_file(path)` | Low-level parser → `dict[str, Any]` |
| `parse_test(path)` | Thin wrapper for `.test` files |
| `parse_solution(path)` | Thin wrapper for `.solution` files |

**Supported formats:**

| Format | Example |
|---|---|
| Single-line quoted | `grid: "base_camp"` |
| Triple-quoted (same line) | `short: """tiny"""` |
| Triple-quoted (multi-line) | `msg: """line 1↵line 2"""` |
| Bare (unquoted) | `iterations: 100` |
| Comments | `# this line is ignored` |
| Blank lines | (ignored) |

CRLF (`\r\n`) is normalised to LF before parsing. Malformed
lines and unterminated blocks raise `ValueError` with the file
path and line number.

---

### test_classes.py (197 lines)

Abstract base classes for questions and test cases.

**`TestCase`** — a single test within a question:

| Method | Description |
|---|---|
| `execute(grades, module_dict, solution_dict)` | Abstract — run the test; must call `test_pass` or `test_fail` |
| `write_solution(module_dict, file_path)` | Write reference data to a `.solution` file (default: no-op) |
| `test_pass(grades)` | Record ✅ pass, return `True` |
| `test_fail(grades, message)` | Record ❌ fail with detail, return `False` |

**`Question`** — a scored group of test cases:

| Method | Description |
|---|---|
| `add_test_case(tc)` | Append a test case |
| `execute(grades, module_dict, solution_dicts)` | Abstract — run all tests and assign points |
| `_run_tests(grades, module_dict, solution_dicts)` | Run each test inside `try/except`; catches `NotImplementedError` (student stubs) and arbitrary exceptions; captures stderr to suppress `raise_not_defined` noise; returns `(passed, total)` |

**Scoring strategies:**

| Class | Rule |
|---|---|
| `PassAllTestsQuestion` | Full credit if ALL tests pass; 0 if any fail |
| `NumberPassedQuestion` | `max_points × passed // total` (integer floor, proportional) |

---

### grading.py (197 lines)

The grade-book that accumulates points and messages.

**`Grades(project_name, questions_and_maxes)`**

| Method | Description |
|---|---|
| `start_question(name)` | Set the active question for subsequent calls |
| `add_points(n)` | Add `n` to the active question (capped at max) |
| `deduct_points(n)` | Remove `n` (floor at 0) |
| `assign_full_credit(n=None)` | Set to `n` or the question's max |
| `fail(message)` | Zero the active question and log the message |
| `add_message(message)` | Append to the active question's log |
| `get_total()` | Return `(total_earned, total_possible)` |
| `produce_output()` | Format and print the per-question table; return the report string |
| `grade(grade_fn, exception_map)` | Run a callable under exception isolation |

Calling any point/message method without a prior `start_question`
raises `RuntimeError`.

---

### mars_test_classes.py (514 lines)

Project-specific `TestCase` subclasses — one per question category.

#### ValueIterationTest (Q1)

- Parses `grid`, `discount`, `noise`, `living_reward`, `iterations`,
  `tolerance` from the `.test` dict.
- Builds a `MarsGrid` with those parameters.
- Constructs the student's `ValueIterationAgent`.
- Collects `V(s)` and `Q(s, a)` at every non-terminal state.
- Compares against the `.solution` file using `compare_pretty_values`
  with the given tolerance (default 0.01).
- `write_solution` runs the reference agent and writes `values:` and
  `q_values:` blocks in pretty-print format.

#### GridPolicyTest (Q2)

- Parses `analysis_fn` (e.g. `"question2a"`) and `expected_policy`
  (a space-separated grid of direction tokens: `N/S/E/W/X`, `_` =
  don't-care).
- Calls the student's analysis function, validates the returned tuple.
- Builds the canyon grid with those parameters, runs VI.
- Checks the resulting policy at every non-underscore cell.
- No `write_solution` (policies are validated by construction, not
  by comparison against stored data).

#### QLearningTest (Q3 / Q5)

- Parses `grid`, `discount`, `noise`, `living_reward`, `num_episodes`,
  `epsilon`, `alpha`, `seed`, `tolerance`.
- Sets `random.seed(seed)` before training for reproducibility.
- Trains the student's `QLearningAgent` using `_train_agent` (which
  calls `start_episode` / `observe_transition` / `stop_episode`).
- Compares learned Q-values against the `.solution` file; falls back
  to a "something was learned" check (non-empty Q-table) if the
  solution is empty.
- `write_solution` trains and dumps Q-values.

#### EpsilonGreedyTest (Q4)

- Parses `epsilon`, `seed`, `num_samples`, `dominant_action`,
  `dominant_q`, `min_rate`, `max_rate`, `num_actions`.
- Constructs a `QLearningAgent` with a fixed action space.
- Manually sets `Q[(state, dominant_action)] = dominant_q`.
- Samples `num_samples` actions with the fixed seed.
- Asserts the empirical frequency of `dominant_action` falls within
  `[min_rate, max_rate]`.
- No `write_solution` (rate check, no reference data needed).

#### Shared helpers

| Function | Description |
|---|---|
| `pretty_print_values(mdp, values)` | Format V-dict as `(x, y): +0.1234` lines |
| `pretty_print_q_values(mdp, q_values)` | Format Q-dict as `(x, y) action: +0.1234` lines |
| `parse_pretty_values(text)` | Inverse — parse text back into a dict. Handles tuple keys `(0, 1)`, Q-value keys `(0, 1) north`, and plain string keys `bias`. |
| `compare_pretty_values(student, reference, tolerance)` | Compare two dicts within tolerance → `(bool, mismatch_message)` |
| `_train_agent(agent, env, num_episodes, max_steps)` | Shared training loop |

---

## Adding New Tests

1. **Create a `.test` file** in the appropriate `test_cases/qN/` directory:
   ```
   class: "ValueIterationTest"
   name: "VI on my_grid"
   grid: "my_grid"
   discount: "0.9"
   noise: "0.2"
   living_reward: "0.0"
   iterations: "100"
   tolerance: "0.01"
   ```

2. **Create an empty `.solution` file** with the same name:
   ```bash
   touch test_cases/q1/vi_my_grid.solution
   ```

3. **Populate it from your reference implementation** (instructor
   workflow; not part of the student release).

4. **Verify:**
   ```bash
   python autograder.py -q q1
   ```

If the new test is for a question type that doesn't store reference
data (Q2 `GridPolicyTest`, Q4 `EpsilonGreedyTest`), the `.solution`
file stays empty — that's by design.

---

## Solution File Formats

**V-values** (`values:` block):
```
(0, 0): 0.6600
(1, 0): 0.7600
(3, 0): 1.0000
```

**Q-values** (`q_values:` block):
```
(0, 0) north: 0.5800
(0, 0) east: 0.6600
(0, 0) south: 0.4400
```

Comparison uses `compare_pretty_values` with tolerance (default
0.01). The `parse_pretty_values` function detects the key format
automatically:

- Ends with `)` → tuple key (V-value)
- Contains `)` followed by a word → Q-value key `((state), action)`

---

## Import Examples

```python
from grading.test_parser import parse_test, parse_solution
from grading.test_classes import (
    Question, TestCase, PassAllTestsQuestion, NumberPassedQuestion,
)
from grading.grading import Grades
from grading.mars_test_classes import (
    ValueIterationTest, GridPolicyTest, QLearningTest,
    EpsilonGreedyTest,
)
```
