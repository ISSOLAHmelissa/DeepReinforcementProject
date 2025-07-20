import numpy as np
from collections import defaultdict

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



import random

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


import numpy as np
import random
from collections import defaultdict

def q_learning_new(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    #  Détection du type d'action
    sample_action = env.available_actions()[0]
    actions_are_int = isinstance(sample_action, int)

    #  Initialisation de Q-table
    if actions_are_int:
        Q = defaultdict(lambda: np.zeros(env.num_actions()))
    else:
        Q = defaultdict(lambda: defaultdict(float))

    for episode in range(num_episodes):
        env.reset()
        s = env.state_id()

        while not env.is_game_over():
            actions = env.available_actions()

            #  Choix de l'action : ε-greedy
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                q_values = Q[s]
                if actions_are_int:
                    valid_q = [(a_, q_values[a_]) for a_ in actions]
                else:
                    valid_q = [(a_, q_values[a_]) for a_ in actions]
                a = max(valid_q, key=lambda x: x[1])[0]

            env.step(a)
            r = env.score()
            s_ = env.state_id()
            done = env.is_game_over()

            #  Mise à jour Q-learning
            if not done:
                next_actions = env.available_actions()
                if actions_are_int:
                    next_max = max(Q[s_][a_] for a_ in next_actions)
                else:
                    next_max = max(Q[s_][a_] for a_ in next_actions)
            else:
                next_max = 0.0

            if actions_are_int:
                Q[s][a] += alpha * (r + gamma * next_max - Q[s][a])
            else:
                Q[s][a] += alpha * (r + gamma * next_max - Q[s][a])

            s = s_

    #  Extraction de la politique optimale
    policy = {}
    for s in Q:
        if actions_are_int:
            policy[s] = int(np.argmax(Q[s]))
        else:
            policy[s] = max(Q[s], key=Q[s].get)

    return policy, Q

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

