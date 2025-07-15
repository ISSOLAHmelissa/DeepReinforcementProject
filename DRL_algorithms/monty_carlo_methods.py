from types import LambdaType
from DRL_algorithms.utils import generate_episode

from collections import defaultdict

def on_policy_fisrt_visit_mc_control(env, epsilon, num_episodes):
    Q = defaultdict(lambda: defaultdict(float))
    Returns = defaultdict(list)
    policy = defaultdict(lambda: defaultdict(float))
    gamma = 0.99
    
    for _ in range(num_episodes):
        episode = generate_episode(env, policy)
        G = 0
        
        # Process episode in reverse
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gamma * G + reward
            
            # First-visit check
            if not any(s == state and a == action for s, a, _ in episode[:i]):
                Returns[(state, action)].append(G)
                Q[state][action] = sum(Returns[(state, action)]) / len(Returns[(state, action)])
                
                # Initialize policy for this state if needed
                if not policy[state]:
                    valid_actions = env.get_valid_actions(state)
                    prob = 1.0 / len(valid_actions)
                    for a in valid_actions:
                        policy[state][a] = prob
                
                # ε-greedy policy update
                max_action = max(Q[state].items(), key=lambda x: x[1])[0]
                for a in policy[state]:
                    if a == max_action:
                        policy[state][a] = 1 - epsilon + (epsilon / len(policy[state]))
                    else:
                        policy[state][a] = epsilon / len(policy[state])
    
    return dict(policy), {k: dict(v) for k, v in Q.items()}

def off_policy_mc_control(env, num_episodes):
    gamma = 0.99
    
    # Initialize data structures with defaultdict
    Q = defaultdict(lambda: defaultdict(float))
    C = defaultdict(lambda: defaultdict(float))
    target_policy = defaultdict(int)  # Stores the greedy action for each state
    behavior_policy = defaultdict(lambda: defaultdict(float))
    
    for episode_num in range(num_episodes):
        # Generate episode using behavior policy (initialized on first access)
        episode = generate_episode(env, behavior_policy)
        G = 0.0
        W = 1.0
        
        # Process episode in reverse
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gamma * G + reward
            
            # Initialize behavior policy for this state if needed
            if not behavior_policy[state]:
                valid_actions = env.get_valid_actions(state)
                prob = 1.0 / len(valid_actions)
                for a in valid_actions:
                    behavior_policy[state][a] = prob
            
            # Skip if action is invalid (shouldn't happen with proper episode generation)
            if action not in behavior_policy[state]:
                continue
                
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            
            # Update target policy greedily
            if Q[state]:  # Only if we have Q-values for this state
                target_policy[state] = max(Q[state].items(), key=lambda x: x[1])[0]
            
            # Early exit if action doesn't match target policy
            if action != target_policy[state]:
                break
                
            # Update importance sampling ratio
            W = W / behavior_policy[state][action]
            if W == 0:
                break
    
    # Convert defaultdicts to regular dicts for the return
    return dict(target_policy), {k: dict(v) for k, v in Q.items()}
