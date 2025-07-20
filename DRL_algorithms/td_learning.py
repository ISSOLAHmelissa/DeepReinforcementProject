import numpy as np
from collections import defaultdict

def sarsa_control(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))
    
    def get_policy_probs(state, available_actions):
        """Epsilon-greedy probs given available actions in a state."""
        if not available_actions:
            return {}
        best_action = max(available_actions, key=lambda a: Q[state][a])
        probs = {a: epsilon / len(available_actions) for a in available_actions}
        probs[best_action] += 1.0 - epsilon
        return probs

    for _ in range(num_episodes):
        env.reset()
        state = env.state
        prev_score = env.score()
        done = env.is_game_over()

        available_actions = env.available_actions()
        action_probs = get_policy_probs(state, available_actions)
        if not action_probs:
            continue
        actions, probs = zip(*action_probs.items())
        action = np.random.choice(actions, p=probs)

        while not done:
            env.step(action)
            new_score = env.score()
            reward = new_score - prev_score
            prev_score = new_score
            next_state = env.state
            done = env.is_game_over()

            next_action = None
            if not done:
                next_available_actions = env.available_actions()
                next_action_probs = get_policy_probs(next_state, next_available_actions)
                if next_action_probs:
                    next_actions, next_probs = zip(*next_action_probs.items())
                    next_action = np.random.choice(next_actions, p=next_probs)

            # SARSA update using the actual next action (on-policy)
            target = reward
            if next_action is not None:
                target += gamma * Q[next_state][next_action]

            Q[state][action] += alpha * (target - Q[state][action])

            state = next_state
            action = next_action

    # Build final greedy policy
    policy = {}
    for state in Q.keys():
        best_action = max(Q[state].items(), key=lambda x: x[1])[0]
        policy[state] = best_action

    # Convert nested defaultdict to normal dict
    Q_dict = {s: dict(aq) for s, aq in Q.items()}

    return policy, Q_dict

import random


def q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Q[state][action] = value
    Q = defaultdict(lambda: dict())
    
    for episode in range(num_episodes):
        env.reset()
        s = env.state_id()

        while not env.is_game_over():
            actions = env.available_actions()
            
            # Initialize Q-values for unseen state-action pairs
            for a_ in actions:
                if a_ not in Q[s]:
                    Q[s][a_] = 0.0

            # Epsilon-greedy policy
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                a = max(Q[s], key=Q[s].get)

            # Step in environment
            next_state, reward, done = env.step(a)
            s_ = env.state_id()

            # Initialize Q-values for next state
            for a_ in env.available_actions():
                if a_ not in Q[s_]:
                    Q[s_][a_] = 0.0

            # Q-learning update
            next_max = max(Q[s_].values()) if not done else 0.0
            Q[s][a] += alpha * (reward + gamma * next_max - Q[s][a])

            # Move to next state
            s = s_

    # Derive greedy policy
    policy = {}
    for state in Q:
        if Q[state]:
            policy[state] = max(Q[state], key=Q[state].get)

    return policy, Q