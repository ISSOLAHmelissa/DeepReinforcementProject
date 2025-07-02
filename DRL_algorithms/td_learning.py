import numpy as np

def expected_sarsa_control(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    all_states = env.get_all_states()

    # 🔹 Initialize Q(s, a) using only valid (state, action) pairs from env
    Q = {}
    for state in all_states:
        Q[state] = {}
        for action in env.get_valid_actions(state):
            Q[state][action] = 0.0

    def get_policy_probs(state):
        """Epsilon-greedy policy probabilities for VALID actions only."""
        valid_actions = list(Q[state].keys())
        best_action = max(valid_actions, key=lambda a: Q[state][a])
        
        probs = {}
        for action in valid_actions:
            probs[action] = epsilon / len(valid_actions)
        probs[best_action] += 1.0 - epsilon
        return probs

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # 🔹 Choose action using epsilon-greedy policy
            action_probs = get_policy_probs(state)
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action = np.random.choice(actions, p=probs)
            
            next_state, reward, done = env.step(action)

            # 🔹 Calculate expected Q-value for next_state
            expected_q = 0.0
            if not done and next_state in Q:
                next_action_probs = get_policy_probs(next_state)
                expected_q = sum(
                    next_action_probs[a] * Q[next_state][a] 
                    for a in Q[next_state]
                )

            # 🔹 Update Q
            Q[state][action] += alpha * (reward + gamma * expected_q - Q[state][action])
            state = next_state

    # 🔹 Derive final greedy policy
    policy = {}
    for state in all_states:
        if Q[state]:
            policy[state] = max(Q[state], key=Q[state].get)
        else:
            policy[state] = None

    return policy, Q
