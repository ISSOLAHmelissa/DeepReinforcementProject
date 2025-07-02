from types import LambdaType
from DRL_algorithms.utils import generate_episode

def on_policy_fisrt_visit_mc_control(env, epsilon, num_episodes):
    all_states = env.get_all_states()
    all_actions = env.get_all_actions()
    gamma = 0.99

    # Initialize Q and policy, but only for valid (state, action) pairs
    Q = {}
    policy = {}
    for state in all_states:
        Q[state] = {}
        policy[state] = {}
        valid_actions = []
        valid_actions = env.get_valid_actions(state)

        # Initialize Q and policy only for valid actions
        for action in valid_actions:
            Q[state][action] = 0.0
            policy[state][action] = 1.0 / len(valid_actions)  # Uniform initial policy
            
        Returns = {}
        for state in all_states:
            valid_actions = env.get_valid_actions(state)
            for action in valid_actions:
                Returns[(state, action)] = []

    for episode_num in range(num_episodes):
        episode = generate_episode(env, policy)
        G = 0
        i = len(episode) - 1
        while i >= 0:
            state, action, reward = episode[i]
            G = gamma * G + reward

            # First-visit check
            if (state, action) not in [(ep[0], ep[1]) for ep in episode[0:i]]:
                Returns[(state, action)].append(G)
                Q[state][action] = sum(Returns[(state, action)]) / len(Returns[(state, action)])

                # Update policy (ε-greedy)
                max_action = max(Q[state], key=Q[state].get)
                for a in policy[state]:
                    if a == max_action:
                        policy[state][a] = 1 - epsilon + (epsilon / len(policy[state]))
                    else:
                        policy[state][a] = epsilon / len(policy[state])
            i -= 1

    return policy, Q

def off_policy_mc_control(env, num_episodes):
    all_states = env.get_all_states()
    all_actions = env.get_all_actions()
    gamma = 0.99

    def get_valid_actions(state):
        row, col = divmod(state, 5)
        valid = []
        valid = env.get_valid_actions(state)
        return valid

    target_policy = {}
    behavior_policy = {}
    Q = {}
    C = {}

    for state in all_states:
        valid_actions = get_valid_actions(state)

        # Initialize greedy target policy arbitrarily
        target_policy[state] = valid_actions[0]

        # Initialize behavior policy with uniform distribution over valid actions
        behavior_policy[state] = {}
        Q[state] = {}
        C[state] = {}
        for action in valid_actions:
            behavior_policy[state][action] = 1.0 / len(valid_actions)
            Q[state][action] = 0.0
            C[state][action] = 0.0

    for episode_num in range(num_episodes):
        episode = generate_episode(env, behavior_policy)

        G = 0.0
        W = 1.0

        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gamma * G + reward

            if action not in Q[state]:
                continue  # skip invalid actions just in case

            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            # Update target policy greedily
            best_action = max(Q[state], key=Q[state].get)
            target_policy[state] = best_action

            if action != best_action:
                break

            prob_b = behavior_policy[state].get(action, 0)
            if prob_b == 0:
                break
            W = W / prob_b

    return target_policy, Q

