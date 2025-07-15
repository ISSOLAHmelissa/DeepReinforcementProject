import numpy as np
from collections import defaultdict

import numpy as np

def expected_sarsa_control(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Initialize Q-table with defaultdict (no need to pre-populate)
    Q = defaultdict(lambda: defaultdict(float))

    def get_policy_probs(state):
        """Epsilon-greedy policy probabilities for valid actions."""
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            return {}
            
        # Find best action (automatically handles new states via defaultdict)
        best_action = max(valid_actions, key=lambda a: Q[state][a])
        
        # Initialize probabilities
        probs = {a: epsilon/len(valid_actions) for a in valid_actions}
        probs[best_action] += 1.0 - epsilon
        return probs

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Get action probabilities
            action_probs = get_policy_probs(state)
            if not action_probs:  # No valid actions (terminal state)
                break
            
            # Choose action
            actions, probs = zip(*action_probs.items())
            action = np.random.choice(actions, p=probs)
            
            # Take action
            next_state, reward, done = env.step(action)

            # Calculate expected Q-value
            expected_q = 0.0
            if not done:
                next_probs = get_policy_probs(next_state)
                expected_q = sum(prob * Q[next_state][a] for a, prob in next_probs.items())

            # Update Q-value (no KeyError thanks to defaultdict)
            Q[state][action] += alpha * (reward + gamma * expected_q - Q[state][action])
            state = next_state

    # Extract policy (convert defaultdict to regular dict)
    policy = {
        state: max(Q[state].items(), key=lambda x: x[1])[0] if Q[state] else None
        for state in set(env.get_all_states()) | set(Q.keys())
    }

    return dict(policy), dict(Q)