import numpy as np
from collections import defaultdict

def expected_sarsa_control(env, num_episodes: int, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))
    
    def get_policy_probs(state, available_actions):
        """Epsilon-greedy policy with available actions."""
        if len(available_actions) == 0:
            return {}
        q_vals = [Q[state][a] for a in available_actions]
        best_action = available_actions[np.argmax(q_vals)]
        probs = {a: epsilon / len(available_actions) for a in available_actions}
        probs[best_action] += 1.0 - epsilon
        return probs

    for _ in range(num_episodes):
        env.reset()
        state = env.state_id()
        prev_score = env.score()
        done = env.is_game_over()
        
        while not done:
            available_actions = env.available_actions()
            # Filter forbidden actions
            available_actions = np.array([a for a in available_actions if not env.is_forbidden(a)])
            if len(available_actions) == 0:
                break

            action_probs = get_policy_probs(state, available_actions)
            actions, probs = zip(*action_probs.items())
            action = np.random.choice(actions, p=probs)

            # Take action
            env.step(action)
            new_score = env.score()
            reward = new_score - prev_score
            prev_score = new_score

            next_state = env.state_id()
            done = env.is_game_over()

            # Expected Q
            expected_q = 0.0
            if not done:
                next_available_actions = env.available_actions()
                next_available_actions = np.array([a for a in next_available_actions if not env.is_forbidden(a)])
                next_action_probs = get_policy_probs(next_state, next_available_actions)
                expected_q = sum(p * Q[next_state][a] for a, p in next_action_probs.items())

            # Q-value update
            Q[state][action] += alpha * (reward + gamma * expected_q - Q[state][action])
            state = next_state

    # Derive greedy policy
    policy = {}
    for state in Q.keys():
        best_action = max(Q[state].items(), key=lambda x: x[1])[0]
        policy[state] = best_action

    # Convert to regular dict
    Q_dict = {s: dict(aq) for s, aq in Q.items()}
    return policy, Q_dict


import random


def q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.num_actions()))
    
    for episode in range(num_episodes):
        env.reset()
        s = env.state_id()
        
        while not env.is_game_over():
            actions = env.available_actions()
            
            # Choix de l'action : ε-greedy
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                q_values = Q[s]
                valid_q = [(a_, q_values[a_]) for a_ in actions]
                a = max(valid_q, key=lambda x: x[1])[0]

            env.step(a)
            r = env.score()  # ou une fonction reward() si différente
            s_ = env.state_id()
            done = env.is_game_over()
            
            # Q-learning update
            if not done:
                next_max = max(Q[s_][a_] for a_ in env.available_actions())
            else:
                next_max = 0.0
            
            Q[s][a] += alpha * (r + gamma * next_max - Q[s][a])
            
            s = s_

    # Deriver la politique optimale à partir de Q
    policy = {}
    for s in Q:
        best_action = np.argmax(Q[s])
        policy[s] = best_action

    return policy, Q
