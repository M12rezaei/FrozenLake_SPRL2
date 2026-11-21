import unittest

import numpy as np

from groupA import sarsa


class TestEnv:
    """A two-step episode with a reward on the final step."""

    n_states = 2
    n_actions = 4

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state == 0:
            self.state = 1
            return 1, 0.0, False

        return 1, 1.0, True


class ExplorationEnv(TestEnv):
    n_actions = 2

    def __init__(self):
        self.episode = 0
        self.visits = []

    def reset(self):
        self.episode += 1
        self.state = 1 if self.episode == 1 else 0
        return self.state

    def step(self, action):
        self.visits.append((self.episode, self.state, action))
        if self.state == 1:
            return 1, float(action == 0), True

        return super().step(action)


class SarsaTests(unittest.TestCase):
    def test_terminal_reward_updates_q(self):
        policy, value = sarsa(
            TestEnv(), max_episodes=1, eta=0.5, gamma=0.9, epsilon=0, seed=0
        )
        np.testing.assert_array_equal(policy, [0, 0])
        np.testing.assert_allclose(value, [0.0, 0.5])

    def test_reward_propagates_to_the_previous_state(self):
        policy, value = sarsa(
            TestEnv(), max_episodes=3, eta=0.5, gamma=0.9, epsilon=0, seed=0
        )
        np.testing.assert_array_equal(policy, [0, 0])
        np.testing.assert_allclose(value, [0.1125, 0.625])

    def test_update_uses_the_exploratory_action(self):
        env = ExplorationEnv()
        _, value = sarsa(
            env, max_episodes=3, eta=0.5, gamma=0.9, epsilon=1, seed=1
        )
        # In episode 2, exploration chooses action 1 even though action 0 is better.
        self.assertEqual(env.visits[:3], [(1, 1, 0), (2, 0, 0), (2, 1, 1)])
        np.testing.assert_allclose(value, [0.0, 0.5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
