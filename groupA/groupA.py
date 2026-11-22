import numpy as np
import contextlib

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.random_state = np.random.RandomState(seed)
        
    def p(self, next_state, state, action):
        raise NotImplementedError()
    
    def r(self, next_state, state, action):
        raise NotImplementedError()
        
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward

        
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, dist, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self.max_steps = max_steps
        
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1./n_states)
        
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)
        
        return self.state
        
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done
    
    def render(self, policy=None, value=None):
        raise NotImplementedError()

        
class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        dist = np.zeros(n_states, dtype=float)
        dist[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        # TODO:
        
        Environment.__init__(self, n_states, n_actions, max_steps, dist, seed=seed)
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        # TODO:
        raise NotImplementedError
    
    def r(self, next_state, state, action):
        # TODO:
        raise NotImplementedError
   
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))
        
def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)

    # TODO:
    for _ in range(max_iterations):
        delta = 0

        for s in range(env.n_states):
            old_value = value[s]
            action = policy[s]

            new_value = 0.0

            for transition in env.P[s][action]:
                probability, next_state, reward, terminated = transition

                future_value = 0.0 if terminated else gamma * value[next_state]

                new_value += probability * (reward + future_value)

            value[s] = new_value

            delta = max(delta, abs(old_value - value[s]))

        if delta < theta:
            break

    return value
    
def policy_improvement(env, policy, value, gamma):
    improved_policy = np.zeros(env.n_states, dtype=int)

    # TODO:
    for s in range(env.n_states):
        action_values = np.zeros(env.n_actions)

        for a in range(env.n_actions):
            for transition in env.P[s][a]:
                probability, next_state, reward, terminated = transition

                future_value = 0.0 if terminated else gamma * value[next_state]

                action_values[a] += probability * (reward + future_value)

        improved_policy[s] = np.argmax(action_values)

    return improved_policy


    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    # TODO:
    value = np.zeros(env.n_states, dtype=float)

    for _ in range(max_iterations):
        # Evaluate the current policy
        value = policy_evaluation(
            env, policy, gamma, theta, max_iterations
        )

        # Improve the current policy
        new_policy = policy_improvement(env, policy, value, gamma)

        # Stop if the policy no longer changes
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=float)
    
    # TODO:
        for _ in range(max_iterations):
            delta = 0

            for s in range(env.n_states):
                old_value = value[s]

                action_values = np.zeros(env.n_actions)

                for a in range(env.n_actions):
                    for transition in env.P[s][a]:
                        probability, next_state, reward, terminated = transition

                        future_value = 0.0 if terminated else gamma * value[next_state]

                        action_values[a] += probability * (reward + future_value)

                value[s] = np.max(action_values)

                delta = max(delta, abs(old_value - value[s]))

            if delta < theta:
                break

    # Extract the greedy policy from the converged value function
    policy = policy_improvement(env, np.zeros(env.n_states, dtype=int), value, gamma)

    return policy, value


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        # using epsilon-greedy policy to choose first action
        if random_state.rand() < epsilon[i]:
            a = random_state.randint(env.n_actions)
        else:
            a = q[s].argmax()

        done = False
        while not done: # not at terminating state 
            next_s, reward, done = env.step(a)
            one_step_return = reward
            if not done: 
                if random_state.rand() < epsilon[i]:
                    next_a = random_state.randint(env.n_actions)
                else:
                    next_a = q[next_s].argmax()

                one_step_return += gamma * q[next_s, next_a]

            td_error = one_step_return - q[s, a]
            q[s, a] += eta[i] * td_error

            if not done:
                s, a = next_s, next_a
            
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:

        done = False
        while not done:
            # using epsilon-greedy policy to pick action
            if random_state.rand() < epsilon[i]:
                a = random_state.randint(env.n_actions)
            else:
                a = q[s].argmax()

            next_s, reward, done = env.step(a)

            one_step_return = reward
            if not done:
                # calculating one step return using next states highest actions value
                one_step_return += gamma * q[next_s].max()

            td_error = one_step_return - q[s, a]
            q[s, a] += eta[i] * td_error

            s = next_s
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value