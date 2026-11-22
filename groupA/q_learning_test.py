import unittest

import numpy as np

from groupA import q_learning
from sarsa_test import ExplorationEnv, TestEnv


class QLearningTests(unittest.TestCase):
    def test_final_step_learns_from_reward(self):
        policy, value = q_learning(
            TestEnv(), max_episodes=1, eta=0.5, gamma=0.9, epsilon=0, seed=0
        )
        np.testing.assert_array_equal(policy, [0, 0])
        np.testing.assert_allclose(value, [0.0, 0.5])

    def test_later_reward_reaches_earlier_state(self):
        policy, value = q_learning(
            TestEnv(), max_episodes=3, eta=0.5, gamma=0.9, epsilon=0, seed=0
        )
        np.testing.assert_array_equal(policy, [0, 0])
        np.testing.assert_allclose(value, [0.1125, 0.625])

    def test_update_uses_best_next_value_during_exploration(self):
        env = ExplorationEnv()
        policy, value = q_learning(
            env, max_episodes=3, eta=0.5, gamma=0.9, epsilon=1, seed=1
        )

        # Episode 2 takes action 1 at state 1. The update for state 0
        # uses action 0's value: 0.25 * 0.9 * 0.5 = 0.1125.
        self.assertEqual(env.visits[:3], [(1, 1, 0), (2, 0, 0), (2, 1, 1)])
        np.testing.assert_array_equal(policy, [0, 0])
        np.testing.assert_allclose(value, [0.1125, 0.5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
