import numpy as np

def expected_sarsa_control(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    all_states = env.get_all_states()
    all_actions = env.get_all_actions()

    # 🔹 Initialize Q(s, a) only for valid (state, action) pairs
    Q = {}
    for state in all_states:
        Q[state] = {}
        row, col = divmod(state, 5)
        if col > 0: Q[state][0] = 0.0  # Left
        if col < 4: Q[state][1] = 0.0  # Right
        if row > 0: Q[state][2] = 0.0  # Up
        if row < 4: Q[state][3] = 0.0  # Down

    def get_policy_probs(state):
        """Epsilon-greedy policy probabilities for VALID actions only."""
        valid_actions = list(Q[state].keys())
        best_action = max(valid_actions, key=lambda a: Q[state][a])
        
        probs = {}
        for action in valid_actions:
            probs[action] = epsilon / len(valid_actions)
        probs[best_action] += 1.0 - epsilon
        return probs

    for episode_num in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # 🔹 Choose action using epsilon-greedy policy
            action_probs = get_policy_probs(state)
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action = np.random.choice(actions, p=probs)
            
            # 🔹 Take step (remove `_` since env.step() returns only 3 values)
            next_state, reward, done = env.step(action)

            # 🔹 Calculate expected Q-value for next_state
            if not done:
                next_action_probs = get_policy_probs(next_state)
                expected_q = sum(
                    next_action_probs[a] * Q[next_state][a] 
                    for a in Q[next_state]  # Only valid actions
                )
            else:
                expected_q = 0.0  # Terminal state has no future rewards

            # 🔹 Update Q-value
            Q[state][action] += alpha * (
                reward + gamma * expected_q - Q[state][action]
            )

            state = next_state

    # 🔹 Derive final greedy policy
    policy = {}
    for state in all_states:
        if Q[state]:  # Only if state has valid actions
            policy[state] = max(Q[state], key=Q[state].get)
        else:
            policy[state] = None  # Handle terminal states if needed

    return policy, Q