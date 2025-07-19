import numpy as np
from collections import defaultdict

from collections import defaultdict
import numpy as np

def sarsa(env, num_episodes: int, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))

    def epsilon_greedy_action(state, available_actions):
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.choice(available_actions)
        q_vals = [Q[state][a] for a in available_actions]
        return available_actions[np.argmax(q_vals)]

    for _ in range(num_episodes):
        env.reset()
        state = env.state_id()
        prev_score = env.score()
        done = env.is_game_over()
        
        available_actions = env.available_actions()
        available_actions = np.array([a for a in available_actions if not env.is_forbidden(a)])
        if len(available_actions) == 0:
            continue
        action = epsilon_greedy_action(state, available_actions)

        while not done:
            env.step(action)
            new_score = env.score()
            reward = new_score - prev_score
            prev_score = new_score
            next_state = env.state_id()
            done = env.is_game_over()

            next_available = env.available_actions()
            next_available = np.array([a for a in next_available if not env.is_forbidden(a)])
            if len(next_available) == 0:
                Q[state][action] += alpha * (reward - Q[state][action])
                break
            next_action = epsilon_greedy_action(next_state, next_available)

            # SARSA update
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action

    # Derive greedy policy
    policy = {}
    for state in Q:
        best_action = max(Q[state].items(), key=lambda x: x[1])[0]
        policy[state] = best_action

    # Convert to regular dict
    Q_dict = {s: dict(aq) for s, aq in Q.items()}
    return policy, Q_dict


from collections import defaultdict
import random

def q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: dict())

    for episode in range(num_episodes):
        env.reset()
        s = env.state_id()
        prev_score = env.score()

        while not env.is_game_over():
            actions = env.available_actions()
            actions = [a for a in actions if not env.is_forbidden(a)]
            if len(actions) == 0:
                break

            # Initialize Q-values
            for a in actions:
                if a not in Q[s]:
                    Q[s][a] = 0.0

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                a = max(Q[s], key=Q[s].get)

            env.step(a)
            s_ = env.state_id()
            reward = env.score() - prev_score
            prev_score = env.score()
            done = env.is_game_over()

            next_actions = env.available_actions()
            next_actions = [a_ for a_ in next_actions if not env.is_forbidden(a_)]
            for a_ in next_actions:
                if a_ not in Q[s_]:
                    Q[s_][a_] = 0.0

            max_q = max(Q[s_].values()) if not done and next_actions else 0.0
            Q[s][a] += alpha * (reward + gamma * max_q - Q[s][a])

            s = s_

    # Derive greedy policy
    policy = {}
    for state in Q:
        if Q[state]:
            policy[state] = max(Q[state], key=Q[state].get)

    return policy, Q
