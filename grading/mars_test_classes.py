"""
grading/mars_test_classes.py -- Mars Rover project-specific test cases (3.10+).

Concrete :class:`grading.test_classes.TestCase` subclasses for each
category of graded question in the Mars Rover RL homework:

* :class:`ValueIterationTest` -- Q1: compare V(s) and Q(s,a) against
  a reference ``.solution`` file after running the student's
  :class:`ValueIterationAgent`.
* :class:`GridPolicyTest` -- Q2: run the student's analysis function,
  build a grid, and check the resulting policy against an expected
  compass-direction map.
* :class:`QLearningTest` -- Q3/Q5: train the student's
  :class:`QLearningAgent` for *N* episodes with a fixed seed, then
  compare learned Q-values against a reference.
* :class:`EpsilonGreedyTest` -- Q4: sample many actions from the
  student's ``get_action`` and verify the exploration rate falls
  within an expected range.
"""

import random
from typing import Any

from grading.test_classes import TestCase
from env.mars_grid import (
    MarsGrid, MarsGridEnvironment, build_mars_grid,
    TERMINAL_STATE, NORTH, SOUTH, EAST, WEST, EXTRACT,
)


# ======================================================================
# Helpers shared across test classes
# ======================================================================


def _parse_float(d: dict, key: str, default: float = 0.0) -> float:
    return float(d.get(key, default))


def _parse_int(d: dict, key: str, default: int = 0) -> int:
    return int(d.get(key, default))


def pretty_print_values(mdp: MarsGrid, values: dict, precision: int = 4) -> str:
    """Format a value table as a readable multi-line string.

    Each non-terminal state appears on its own line as
    ``(x, y): +0.1234``. TERMINAL_STATE is omitted (it is
    always 0 by definition).
    """
    lines: list[str] = []
    for state in sorted(s for s in mdp.get_states() if s != TERMINAL_STATE):
        v = values.get(state, 0.0)
        lines.append(f'{state}: {v:.{precision}f}')
    return '\n'.join(lines)


def pretty_print_q_values(mdp: MarsGrid, q_values: dict, precision: int = 4) -> str:
    """Format Q-values as ``(x, y) action: +0.1234`` lines."""
    lines: list[str] = []
    for state in sorted(s for s in mdp.get_states() if s != TERMINAL_STATE):
        for action in mdp.get_possible_actions(state):
            q = q_values.get((state, action), 0.0)
            lines.append(f'{state} {action}: {q:.{precision}f}')
    return '\n'.join(lines)


def parse_pretty_values(text: str) -> dict:
    """Inverse of :func:`pretty_print_values`. Returns ``{state: float}``."""
    result: dict = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        key_part, _, val_part = line.rpartition(':')
        key_part = key_part.strip()
        val = float(val_part.strip())
        if '(' not in key_part:
            # Plain string key (e.g. weight names like 'bias').
            result[key_part] = val
        elif key_part.endswith(')'):
            # Tuple key like "(0, 1)".
            result[eval(key_part)] = val
        else:
            # Q-value form: "(x, y) action".
            paren_end = key_part.rindex(')') + 1
            state = eval(key_part[:paren_end])
            action = key_part[paren_end:].strip()
            result[(state, action)] = val
    return result


def compare_pretty_values(
    student: dict,
    reference: dict,
    tolerance: float = 0.01,
) -> tuple[bool, str]:
    """Compare two value / Q-value dicts within ``tolerance``.

    Returns ``(all_match, first_mismatch_message)``.
    """
    for key in reference:
        ref = reference[key]
        stu = student.get(key, 0.0)
        if abs(ref - stu) > tolerance:
            return False, (
                f'{key}: student={stu:.4f} vs reference={ref:.4f} '
                f'(diff={abs(ref - stu):.6f}, tol={tolerance})'
            )
    return True, ''


# ======================================================================
# Shared training helper
# ======================================================================


def _train_agent(agent: Any, env: MarsGridEnvironment,
                 num_episodes: int, max_steps: int = 200) -> None:
    """Train ``agent`` on ``env`` for ``num_episodes`` full episodes."""
    for _ in range(num_episodes):
        env.reset()
        agent.start_episode()
        for _ in range(max_steps):
            if env.is_terminal():
                break
            state = env.get_current_state()
            action = agent.get_action(state)
            if action is None:
                break
            next_state, reward = env.do_action(action)
            agent.observe_transition(state, action, next_state, reward)
        agent.stop_episode()


# ======================================================================
# Q1: Value Iteration
# ======================================================================


class ValueIterationTest(TestCase):
    """Compare student's VI values/Q-values against a reference solution."""

    def execute(self, grades: Any, module_dict: dict, solution_dict: dict):
        td = self.test_dict
        grid_name = td.get('grid', 'base_camp')
        discount = _parse_float(td, 'discount', 0.9)
        noise = _parse_float(td, 'noise', 0.2)
        living = _parse_float(td, 'living_reward', 0.0)
        iterations = _parse_int(td, 'iterations', 100)
        tolerance = _parse_float(td, 'tolerance', 0.01)

        mdp = build_mars_grid(grid_name, noise=noise,
                              living_reward=living, discount=discount)

        from students.value_iteration_agents import ValueIterationAgent
        agent = ValueIterationAgent(mdp, discount=discount,
                                    iterations=iterations)

        # Build student value / Q-value dicts.
        student_v: dict = {}
        student_q: dict = {}
        for state in mdp.get_states():
            if state == TERMINAL_STATE:
                continue
            student_v[state] = agent.get_value(state)
            for action in mdp.get_possible_actions(state):
                student_q[(state, action)] = agent.get_q_value(state, action)

        # Compare values.
        if 'values' in solution_dict:
            ref_v = parse_pretty_values(solution_dict['values'])
            ok, msg = compare_pretty_values(student_v, ref_v, tolerance)
            if not ok:
                return self.test_fail(grades, f'V mismatch: {msg}')

        # Compare Q-values.
        if 'q_values' in solution_dict:
            ref_q = parse_pretty_values(solution_dict['q_values'])
            ok, msg = compare_pretty_values(student_q, ref_q, tolerance)
            if not ok:
                return self.test_fail(grades, f'Q mismatch: {msg}')

        return self.test_pass(grades)

    def write_solution(self, module_dict: dict, file_path: str) -> None:
        td = self.test_dict
        grid_name = td.get('grid', 'base_camp')
        discount = _parse_float(td, 'discount', 0.9)
        noise = _parse_float(td, 'noise', 0.2)
        living = _parse_float(td, 'living_reward', 0.0)
        iterations = _parse_int(td, 'iterations', 100)

        mdp = build_mars_grid(grid_name, noise=noise,
                              living_reward=living, discount=discount)

        from students.value_iteration_agents import ValueIterationAgent
        agent = ValueIterationAgent(mdp, discount=discount,
                                    iterations=iterations)

        v_dict = {s: agent.get_value(s)
                  for s in mdp.get_states() if s != TERMINAL_STATE}
        q_dict = {(s, a): agent.get_q_value(s, a)
                  for s in mdp.get_states() if s != TERMINAL_STATE
                  for a in mdp.get_possible_actions(s)}

        with open(file_path, 'w') as f:
            f.write(f'values: """\n{pretty_print_values(mdp, v_dict)}\n"""\n')
            f.write(f'q_values: """\n{pretty_print_q_values(mdp, q_dict)}\n"""\n')


# ======================================================================
# Q2: Grid Policy (analysis questions)
# ======================================================================


_DIR_MAP = {'N': NORTH, 'S': SOUTH, 'E': EAST, 'W': WEST, 'X': EXTRACT}


class GridPolicyTest(TestCase):
    """Check that the student's analysis answer produces the expected policy.

    ``test_dict`` must contain:
    * ``analysis_fn`` -- name of the analysis function (e.g. ``question2a``).
    * ``expected_policy`` -- a space-separated grid of direction tokens
      (N/S/E/W/X for extract, _ for don't-care), one row per line,
      top-row-first (matching the grid text convention).
    """

    def execute(self, grades: Any, module_dict: dict, solution_dict: dict):
        td = self.test_dict
        fn_name = td.get('analysis_fn', '')
        expected_raw = td.get('expected_policy', '')
        iterations = _parse_int(td, 'iterations', 100)

        import students.analysis as analysis
        fn = getattr(analysis, fn_name, None)
        if fn is None:
            return self.test_fail(grades, f'{fn_name} not found in analysis')

        answer = fn()
        if answer == 'NOT POSSIBLE':
            return self.test_fail(grades, f'{fn_name}() still returns NOT POSSIBLE')
        if not (isinstance(answer, tuple) and len(answer) == 3):
            return self.test_fail(grades, f'{fn_name}() must return a 3-tuple')

        discount, noise, living = answer
        mdp = build_mars_grid('canyon', noise=noise,
                              living_reward=living, discount=discount)

        from students.value_iteration_agents import ValueIterationAgent
        agent = ValueIterationAgent(mdp, discount=discount,
                                    iterations=iterations)

        # Parse expected policy grid.
        if not expected_raw.strip():
            # No expected policy to check -- just verify the tuple ran VI
            return self.test_pass(grades)

        rows = [row.split() for row in expected_raw.strip().splitlines()]
        height = len(rows)
        width = len(rows[0]) if rows else 0
        for row_idx, row in enumerate(rows):
            y = height - 1 - row_idx
            for x, token in enumerate(row):
                if token == '_':
                    continue
                expected_action = _DIR_MAP.get(token)
                if expected_action is None:
                    continue
                actual = agent.get_policy((x, y))
                if actual != expected_action:
                    return self.test_fail(
                        grades,
                        f'policy at ({x},{y}): expected {expected_action}, '
                        f'got {actual}'
                    )
        return self.test_pass(grades)


# ======================================================================
# Q3 / Q5: Q-Learning
# ======================================================================


class QLearningTest(TestCase):
    """Train QL agent for N episodes with a fixed seed, compare Q-values."""

    def execute(self, grades: Any, module_dict: dict, solution_dict: dict):
        td = self.test_dict
        grid_name = td.get('grid', 'base_camp')
        discount = _parse_float(td, 'discount', 0.9)
        noise = _parse_float(td, 'noise', 0.2)
        living = _parse_float(td, 'living_reward', 0.0)
        num_episodes = _parse_int(td, 'num_episodes', 100)
        epsilon = _parse_float(td, 'epsilon', 0.3)
        alpha = _parse_float(td, 'alpha', 0.5)
        seed = _parse_int(td, 'seed', 42)
        tolerance = _parse_float(td, 'tolerance', 0.1)

        mdp = build_mars_grid(grid_name, noise=noise,
                              living_reward=living, discount=discount)
        env = MarsGridEnvironment(mdp)

        from students.qlearning_agents import QLearningAgent
        agent = QLearningAgent(
            action_fn=env.get_possible_actions,
            num_training=num_episodes,
            epsilon=epsilon, alpha=alpha, gamma=discount,
        )

        random.seed(seed)
        _train_agent(agent, env, num_episodes)

        # Build student Q-value dict from the trained agent.
        student_q: dict = {}
        for state in mdp.get_states():
            if state == TERMINAL_STATE:
                continue
            for action in mdp.get_possible_actions(state):
                q = agent.get_q_value(state, action)
                if q != 0.0:
                    student_q[(state, action)] = q

        if 'q_values' in solution_dict:
            ref_q = parse_pretty_values(solution_dict['q_values'])
            ok, msg = compare_pretty_values(student_q, ref_q, tolerance)
            if not ok:
                return self.test_fail(grades, f'Q mismatch: {msg}')

        # Fallback: just check something was learned.
        if not student_q:
            return self.test_fail(grades, 'no Q-values learned')
        return self.test_pass(grades)

    def write_solution(self, module_dict: dict, file_path: str) -> None:
        td = self.test_dict
        grid_name = td.get('grid', 'base_camp')
        discount = _parse_float(td, 'discount', 0.9)
        noise = _parse_float(td, 'noise', 0.2)
        living = _parse_float(td, 'living_reward', 0.0)
        num_episodes = _parse_int(td, 'num_episodes', 100)
        epsilon = _parse_float(td, 'epsilon', 0.3)
        alpha = _parse_float(td, 'alpha', 0.5)
        seed = _parse_int(td, 'seed', 42)

        mdp = build_mars_grid(grid_name, noise=noise,
                              living_reward=living, discount=discount)
        env = MarsGridEnvironment(mdp)

        from students.qlearning_agents import QLearningAgent
        agent = QLearningAgent(
            action_fn=env.get_possible_actions,
            num_training=num_episodes,
            epsilon=epsilon, alpha=alpha, gamma=discount,
        )
        random.seed(seed)
        _train_agent(agent, env, num_episodes)

        q_dict = {}
        for s in mdp.get_states():
            if s == TERMINAL_STATE:
                continue
            for a in mdp.get_possible_actions(s):
                q = agent.get_q_value(s, a)
                if q != 0.0:
                    q_dict[(s, a)] = q

        with open(file_path, 'w') as f:
            f.write(f'q_values: """\n{pretty_print_q_values(mdp, q_dict)}\n"""\n')


# ======================================================================
# Q4: Epsilon-Greedy
# ======================================================================


class EpsilonGreedyTest(TestCase):
    """Sample many actions and verify exploration rate is in range."""

    def execute(self, grades: Any, module_dict: dict, solution_dict: dict):
        td = self.test_dict
        epsilon = _parse_float(td, 'epsilon', 0.5)
        seed = _parse_int(td, 'seed', 42)
        num_samples = _parse_int(td, 'num_samples', 2000)
        dominant_action = td.get('dominant_action', NORTH)
        dominant_q = _parse_float(td, 'dominant_q', 10.0)
        min_rate = _parse_float(td, 'min_rate', 0.4)
        max_rate = _parse_float(td, 'max_rate', 0.95)
        num_actions = _parse_int(td, 'num_actions', 4)

        actions = (NORTH, SOUTH, EAST, WEST)[:num_actions]

        from students.qlearning_agents import QLearningAgent
        agent = QLearningAgent(
            action_fn=lambda state: actions,
            num_training=num_samples,
            epsilon=epsilon, alpha=0.5, gamma=0.9,
        )
        state = 'epsilon_test_state'
        agent.q_values[(state, dominant_action)] = dominant_q

        random.seed(seed)
        counts: dict[str, int] = {a: 0 for a in actions}
        for _ in range(num_samples):
            a = agent.get_action(state)
            if a in counts:
                counts[a] += 1

        rate = counts[dominant_action] / num_samples
        if rate < min_rate:
            return self.test_fail(
                grades,
                f'P({dominant_action})={rate:.3f} < {min_rate} '
                f'(no exploitation?)'
            )
        if rate > max_rate:
            return self.test_fail(
                grades,
                f'P({dominant_action})={rate:.3f} > {max_rate} '
                f'(no exploration?)'
            )
        return self.test_pass(grades)
