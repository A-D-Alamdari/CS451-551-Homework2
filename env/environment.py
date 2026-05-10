"""
env/environment.py -- abstract interactive environment interface (3.10+).

Part of the "Mars Rover: Red Planet Rescue" homework for CS 451/551.

Where :mod:`env.mdp` describes a planning problem in full -- every
state, every transition probability, every reward -- :class:`Environment`
describes the same world from the agent's *experiential* point of view.
A reinforcement-learning agent does not get to read the transition
table; it picks an action, calls :meth:`do_action`, and receives back
the next state and a scalar reward. That is all the information a
model-free learner such as Q-learning is allowed to use.

Concrete subclasses (the grid-world wrapper around
:class:`env.mars_grid.MarsGrid`, or the 1-D simulator driving the
crawler) wrap an MDP and maintain the current state on behalf of the
agent. This separation lets the same Q-learning implementation drive
both the Mars Rover grid world and the crawler without modification.
"""

from typing import Any


class Environment:
    """Abstract interactive environment the agent steps through.

    Exposes only the operations a model-free learner legitimately
    needs: query the current state, list legal actions, commit to an
    action and receive ``(next_state, reward)``, reset for a new
    episode, and ask whether the current state ends the episode.
    Crucially, there is no method for peeking at transition
    probabilities -- a correct Q-learning agent never needs one.
    """

    def get_current_state(self) -> Any:
        """Return the state the agent currently occupies."""
        raise NotImplementedError

    def get_possible_actions(self, state: Any) -> tuple[Any, ...]:
        """Return the actions legally available from ``state``.

        Should return an empty tuple for a terminal state.
        """
        raise NotImplementedError

    def do_action(self, action: Any) -> tuple[Any, float]:
        """Take ``action`` from the current state and advance.

        Subclasses must sample a successor s' from the environment
        dynamics, compute the scalar reward R(s, a, s'), update the
        internal state to s', and return the ``(next_state, reward)``
        pair so the caller can feed the experience tuple into its
        learning rule.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Return the environment to its start state for a new episode.

        After a reset, :meth:`get_current_state` must return the
        environment's designated initial state and :meth:`is_terminal`
        must return False.
        """
        raise NotImplementedError

    def is_terminal(self) -> bool:
        """Return whether the current state ends the episode.

        Unlike :meth:`env.mdp.MarkovDecisionProcess.is_terminal` this
        takes no argument -- the environment already knows where the
        agent is. The training loop uses this to decide when to call
        :meth:`reset`.
        """
        raise NotImplementedError
