"""
env/mdp.py -- abstract Markov Decision Process interface (Python 3.10+).

Part of the "Mars Rover: Red Planet Rescue" homework for CS 451/551
(Introduction to AI) at Ozyegin University.

A :class:`MarkovDecisionProcess` fully specifies a planning problem for
model-based algorithms such as value iteration: it enumerates states,
exposes legal actions per state, supplies the stochastic transition
model T(s, a, s') = P(s' | s, a), and defines the reward function
R(s, a, s'). Concrete subclasses (for example
:class:`env.mars_grid.MarsGrid`) plug in the specifics of a particular
world; the planners in ``students/value_iteration_agents.py`` consume
the abstract interface so they work unchanged across worlds.

This file is intentionally pure interface: every method raises
``NotImplementedError``. Subclasses must override each method.
"""

from typing import Any


class MarkovDecisionProcess:
    """Abstract description of a fully observable Markov decision process.

    A subclass must supply the five-tuple (S, A, T, R, s0) and a way to
    recognise terminal states. The Mars Rover environment folds the
    20% solar-interference movement noise into
    :meth:`get_transition_states_and_probs`; from the planner's
    perspective the noise is simply part of the model.
    """

    def get_states(self) -> list[Any]:
        """Return a concrete sequence containing every state in the MDP.

        Value iteration sweeps this set repeatedly, so subclasses must
        return a list (or other reusable sequence), not a one-shot
        generator.
        """
        raise NotImplementedError

    def get_start_state(self) -> Any:
        """Return the unique start state s0 for the MDP."""
        raise NotImplementedError

    def get_possible_actions(self, state: Any) -> tuple[Any, ...]:
        """Return a tuple of legal actions available from ``state``.

        Terminal states must return an empty tuple; non-terminal
        states should always return at least one action so the
        planner can make progress.
        """
        raise NotImplementedError

    def get_transition_states_and_probs(
        self, state: Any, action: Any
    ) -> list[tuple[Any, float]]:
        """Return T(s, a, .) as ``[(next_state, probability), ...]``.

        Probabilities across the returned list must be non-negative
        and sum to 1. Deterministic transitions are represented as a
        singleton list with probability 1.0. This method is only
        meaningful for model-based agents; Q-learning agents must
        not call it.
        """
        raise NotImplementedError

    def get_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Return the immediate scalar reward R(s, a, s')."""
        raise NotImplementedError

    def is_terminal(self, state: Any) -> bool:
        """Return whether ``state`` is absorbing.

        :meth:`get_possible_actions` must return an empty tuple for any
        state where this returns True; the planner relies on that
        invariant to anchor the value-iteration recursion.
        """
        raise NotImplementedError
