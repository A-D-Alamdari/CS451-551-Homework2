"""
env/crawler.py -- the prototype crawler environment (text-only, 3.10+).

Part of the "Mars Rover: Red Planet Rescue" homework for CS 451/551.

    Back at the orbital platform above Mars, engineers have set up
    a scaled-down prototype crawler in a simulation chamber: a
    boxy body on wheels with a single two-segment arm
    (shoulder-to-elbow, elbow-to-hand). Unlike Rover-7 on the
    surface, the prototype has no prior model of how its joints
    interact with the chamber floor -- it has to figure out
    locomotion by itself. At each time step it can nudge one of
    its two joints by a single increment. If the action plants the
    hand on the floor and drags it backwards, friction pushes the
    body forward; if the action lifts the hand into the air,
    nothing happens. A Q-learning agent discovering that the
    plant-and-drag cycle is rewarded is exactly what Q3 of the
    homework asks the student to implement.

The domain is small enough to learn in a few thousand steps
(9 arm buckets x 13 hand buckets = 117 discrete states, 4 actions
at most) and tests the student's QLearningAgent against a
*continuing* task with no terminal state -- the opposite of the
Mars grid world, which is shaped around episodes that end at a
reward cell.

This module is intentionally decoupled from any rendering code;
a GUI front-end (such as the flat-layout ``crawler.py`` at the
project root) can wrap :class:`CrawlerEnvironment` later. The
text-mode driver :func:`run_crawler` imports the student's
``QLearningAgent`` from ``students.qlearning_agents`` so no
modification to that file is required to run the crawler -- just
fill in the tabular Q-learning stubs.
"""

import math
from typing import Any

from env.environment import Environment


class CrawlerEnvironment(Environment):
    """A two-segment-arm crawler in a 2-D simulation chamber.

    State is a pair of integer indices ``(arm_index, hand_index)``
    into discretised arrays of joint angles. The arm and hand each
    swing through ``[-pi/3, +pi/3]`` (120 degrees) split into
    ``num_arm_states`` and ``num_hand_states`` buckets respectively.
    At each step the agent picks one of four actions::

        'arm_up'    : arm_index += 1
        'arm_down'  : arm_index -= 1
        'hand_up'   : hand_index += 1
        'hand_down' : hand_index -= 1

    Actions that would leave the joint range are absent from the
    legal action set.

    After updating joint indices the environment runs forward
    kinematics to figure out where the hand tip ended up in world
    coordinates. If the hand is on the chamber floor (within 10
    pixels of ``ground_y``) the body slides forward by half of
    whatever horizontal distance the hand swept backwards, and the
    agent receives a small positive reward proportional to that
    backward swing. If the hand is in the air, no motion and no
    reward. The task is continuing -- :meth:`is_terminal` always
    returns ``False``.

    Attributes
    ----------
    robot_x : float
        The body's world x-coordinate. Increases when the crawler
        successfully crawls forward. Starts at 200.
    arm_index, hand_index : int
        Current joint-angle bucket indices.
    arm_buckets, hand_buckets : list of float
        Precomputed angle values per bucket index.
    """

    def __init__(
        self,
        arm_min: float = -math.pi / 3,
        arm_max: float = math.pi / 3,
        hand_min: float = -math.pi / 3,
        hand_max: float = math.pi / 3,
        num_arm_states: int = 9,
        num_hand_states: int = 13,
        arm_length: float = 60,
        hand_length: float = 40,
        ground_y: float = 300,
        robot_width: float = 40,
        robot_height: float = 20,
    ):
        # Geometry. robot_height=20 places the shoulder at
        # y = ground_y - 20 = 280, which is exactly the top edge
        # of the spec's 40x20 yellow body rect at (robot_x-20, 280);
        # any GUI rendering wrapping this environment gets a body
        # and arm that line up without cheating.
        self.arm_length = arm_length
        self.hand_length = hand_length
        self.ground_y = ground_y
        self.robot_width = robot_width
        self.robot_height = robot_height

        # Discretised angle arrays. Linear spacing from arm_min to
        # arm_max (inclusive) means bucket 0 is the most "down"
        # angle and the last bucket is the most "up".
        self.num_arm_states = num_arm_states
        self.num_hand_states = num_hand_states
        self.arm_buckets: list[float] = [
            arm_min + i * (arm_max - arm_min) / (num_arm_states - 1)
            for i in range(num_arm_states)
        ]
        self.hand_buckets: list[float] = [
            hand_min + i * (hand_max - hand_min) / (num_hand_states - 1)
            for i in range(num_hand_states)
        ]

        # Dynamic state: joint indices and body position. Delegated
        # to reset() so initial and post-reset states stay in sync.
        self.arm_index: int = 0
        self.hand_index: int = 0
        self.robot_x: float = 0.0
        self.reset()

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def get_current_state(self) -> tuple[int, int]:
        return (self.arm_index, self.hand_index)

    def get_possible_actions(
        self, state: tuple[int, int] | None = None
    ) -> tuple[str, ...]:
        """Return legal actions from ``state`` (or current if None).

        The Q-learning agent's ``action_fn`` hook calls this with a
        specific state when it needs to know the legal actions at
        a previously-seen cell; the training loop and
        :meth:`do_action` call it with no argument.
        """
        if state is None:
            arm_index, hand_index = self.arm_index, self.hand_index
        else:
            arm_index, hand_index = state
        actions: list[str] = []
        if arm_index < self.num_arm_states - 1:
            actions.append('arm_up')
        if arm_index > 0:
            actions.append('arm_down')
        if hand_index < self.num_hand_states - 1:
            actions.append('hand_up')
        if hand_index > 0:
            actions.append('hand_down')
        return tuple(actions)

    def do_action(self, action: str) -> tuple[tuple[int, int], float]:
        """Apply ``action``, update body position, return ``(s', r)``."""
        # Step 1: record where the hand is *now* so that after the
        # joint update we can measure how far the hand swept
        # horizontally purely as a result of the joint change.
        old_hand_x = self._compute_hand_x()

        # Step 2: apply the joint update. Illegal actions should
        # have been filtered by get_possible_actions; clamp
        # defensively so a misbehaving caller cannot drive indices
        # out of range.
        if action == 'arm_up':
            self.arm_index = min(self.num_arm_states - 1, self.arm_index + 1)
        elif action == 'arm_down':
            self.arm_index = max(0, self.arm_index - 1)
        elif action == 'hand_up':
            self.hand_index = min(self.num_hand_states - 1, self.hand_index + 1)
        elif action == 'hand_down':
            self.hand_index = max(0, self.hand_index - 1)

        # Step 3: forward kinematics on the *new* joint angles but
        # the *old* body position. This gives the joint-driven
        # sweep of the hand independently of any body motion the
        # sweep is about to cause.
        new_hand_x, new_hand_y = self._compute_hand_position()

        reward = 0.0
        if new_hand_y >= self.ground_y - 10:
            # Hand is on the floor. If the joint change swept the
            # hand backwards (new < old), treating the hand as
            # anchored, the body slides forward by half of that
            # sweep and the agent is rewarded proportionally.
            # Forward sweeps get negative dx -> negative reward,
            # correctly punishing "pushing off" in the wrong
            # direction.
            dx = old_hand_x - new_hand_x
            self.robot_x += dx * 0.5
            reward = dx * 0.1

        return (self.get_current_state(), reward)

    def reset(self) -> None:
        """Return joints to their midpoint and the body to x=200."""
        self.arm_index = self.num_arm_states // 2
        self.hand_index = self.num_hand_states // 2
        self.robot_x = 200.0

    def is_terminal(self) -> bool:
        # Continuing task -- no terminal state.
        return False

    # ------------------------------------------------------------------
    # Forward-kinematics helpers
    # ------------------------------------------------------------------

    def _compute_hand_x(self) -> float:
        """Return just the world x-coordinate of the hand tip."""
        x, _ = self._compute_hand_position()
        return x

    def _compute_hand_position(self) -> tuple[float, float]:
        """Forward kinematics: ``(hand_x, hand_y)`` in world space.

        Screen coordinates have y growing downward, so a positive
        joint angle rotates the segment *upward* and therefore
        *subtracts* from the y of the next joint.
        """
        arm_angle = self.arm_buckets[self.arm_index]
        hand_angle = self.hand_buckets[self.hand_index]

        shoulder_x = self.robot_x + self.robot_width / 2.0
        shoulder_y = self.ground_y - self.robot_height

        elbow_x = shoulder_x + self.arm_length * math.cos(arm_angle)
        elbow_y = shoulder_y - self.arm_length * math.sin(arm_angle)

        # The hand's world angle is the sum of shoulder + elbow
        # joint angles.
        total_angle = arm_angle + hand_angle
        hand_x = elbow_x + self.hand_length * math.cos(total_angle)
        hand_y = elbow_y - self.hand_length * math.sin(total_angle)
        return (hand_x, hand_y)


# ----------------------------------------------------------------------
# Text-mode training driver
# ----------------------------------------------------------------------


def run_crawler(
    num_steps: int = 500,
    epsilon: float = 0.3,
    alpha: float = 0.5,
    gamma: float = 0.9,
    report_every: int = 100,
) -> tuple[Any, CrawlerEnvironment]:
    """Train a Q-learning agent on the crawler for ``num_steps`` steps.

    Pulls in :class:`students.qlearning_agents.QLearningAgent` --
    i.e. the student's tabular Q-learning implementation -- wires
    it to a fresh :class:`CrawlerEnvironment`, and runs a simple
    loop that prints progress every ``report_every`` steps.

    The crawler is a continuing task (no terminal state), so the
    whole run is a single "episode" as far as the agent's
    bookkeeping is concerned. That keeps the training loop
    trivial: pick an action, step the world, learn, repeat.

    Returns the trained ``(agent, env)`` pair so a caller can
    inspect the learned Q-table and the final body position.
    """
    # Lazy import keeps this module importable even if students
    # have not yet touched their qlearning_agents.py stubs.
    from students.qlearning_agents import QLearningAgent

    env = CrawlerEnvironment()
    # Setting num_training = num_steps keeps alpha/epsilon at
    # their specified values for the full run. We use the agent's
    # action_fn hook to let it query legal moves straight from the
    # environment without having to know anything about joint
    # indices.
    agent = QLearningAgent(
        action_fn=env.get_possible_actions,
        num_training=max(num_steps, 1),
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
    )

    print(
        f"Crawler simulation chamber online. Running Q-learning for "
        f"{num_steps} steps (epsilon={epsilon}, alpha={alpha}, gamma={gamma})."
    )

    start_x = env.robot_x
    agent.start_episode()
    for step in range(1, num_steps + 1):
        state = env.get_current_state()
        action = agent.get_action(state)
        if action is None:
            # Should never happen for the crawler (always at least
            # two legal moves); defensive.
            break
        next_state, reward = env.do_action(action)
        agent.observe_transition(state, action, next_state, reward)

        if step % report_every == 0:
            distance = env.robot_x - start_x
            print(
                f"  step {step:5d}/{num_steps}: "
                f"robot_x={env.robot_x:8.2f}  "
                f"distance_crawled={distance:+8.2f}  "
                f"episode_reward={agent.episode_rewards:+8.3f}"
            )
    agent.stop_episode()
    print(f"Training complete. Final robot_x = {env.robot_x:.2f}")
    return agent, env
