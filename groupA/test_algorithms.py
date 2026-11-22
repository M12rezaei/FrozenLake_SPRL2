
import numpy as np

from groupA import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration
)


class TestEnv:
    """A small deterministic MDP for testing RL algorithms."""

    n_states = 3
    n_actions = 2

    # P[state][action] = [(probability, next_state, reward, terminated)]
    P = {
        0: {
            0: [(1.0, 1, 1.0, False)],   # Safe: go to state 1
            1: [(1.0, 0, 5.0, True)],    # Terminate with reward 5
        },
        1: {
            0: [(1.0, 2, 2.0, False)],   # Safe: go to state 2
            1: [(1.0, 0, 1.0, True)],    # Risky: terminate with reward 1
        },
        2: {
            0: [(1.0, 0, 3.0, True)],    # Safe: terminate with reward 3
            1: [(1.0, 0, 10.0, True)],   # Risky: terminate with reward 10
        },
    }


env = TestEnv()

gamma = 0.9
theta = 1e-10
max_iterations = 10_000


# Test 1: Policy evaluation

def test_policy_evaluation():
    # Always choose action 0 (safe path)
    policy = np.array([0, 0, 0])

    value = policy_evaluation(
        env, policy, gamma, theta, max_iterations
    )

    # Expected:
    # V(2) = 3
    # V(1) = 2 + 0.9 * 3 = 4.7
    # V(0) = 1 + 0.9 * 4.7 = 5.23

    expected = np.array([5.23, 4.7, 3.0])

    assert np.allclose(value, expected, atol=1e-6)

    print("Policy evaluation passed!")
    print("Value:", value)

# Test 2: Policy improvement

def test_policy_improvement():
    # Start with safe-path values
    value = np.array([5.23, 4.7, 3.0])

    policy = np.array([0, 0, 0])

    improved_policy = policy_improvement(
        env, policy, value, gamma
    )

    # Expected optimal actions:
    # State 0 -> Action 0 (value 5.23 beats terminal reward 5)
    # State 1 -> Action 0 (value 4.7 beats terminal reward 1)
    # State 2 -> Action 1 (reward 10 beats reward 3)

    expected = np.array([0, 0, 1])

    assert np.array_equal(improved_policy, expected)

    print("Policy improvement passed!")
    print("Improved policy:", improved_policy)



if __name__ == "__main__":
    test_policy_evaluation()
    test_policy_improvement()