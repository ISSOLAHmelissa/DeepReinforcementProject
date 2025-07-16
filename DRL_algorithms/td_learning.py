import numpy as np
from collections import defaultdict

def expected_sarsa_control(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))
    
    def get_policy_probs(state, available_actions):
        """Epsilon-greedy probs given available actions in a state."""
        if not available_actions.size:
            return {}
        best_action = max(available_actions, key=lambda a: Q[state][a])
        probs = {a: epsilon/len(available_actions) for a in available_actions}
        probs[best_action] += 1.0 - epsilon
        return probs
    
    for _ in range(num_episodes):
        env.reset()
        state = env.state_id()
        prev_score = env.score()
        done = env.is_game_over()
        
        while not done:
            available_actions = env.available_actions()
            action_probs = get_policy_probs(state, available_actions)
            if not action_probs:
                break
            
            # Sample action
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
                next_probs = get_policy_probs(next_state, next_available_actions)
                expected_q = sum(p * Q[next_state][a] for a, p in next_probs.items())
            
            # Update Q
            Q[state][action] += alpha * (reward + gamma * expected_q - Q[state][action])
            
            state = next_state
    
    # Build final policy: greedy w.r.t. Q
    policy = {}
    for state in Q.keys():
        best_action = max(Q[state].items(), key=lambda x: x[1])[0]
        policy[state] = best_action

    # Convert nested defaultdict to dict
    Q_dict = {s: dict(aq) for s, aq in Q.items()}
    
    return policy, Q_dict
