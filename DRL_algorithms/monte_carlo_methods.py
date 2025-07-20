import random
from collections import defaultdict

def generate_episode(env, policy):
    episode = []
    env.reset()  # remet l'environnement à zéro
    state = env.state_id()
    done = env.is_game_over()
    previous_score = env.score()
    
    while not done:
        valid_actions = env.available_actions()
        if len(valid_actions) == 0:
            print(f"[WARNING] No valid actions for state: {state}")
            break

        # Sélection d'action
        if state in policy and policy[state]:
            actions = list(policy[state].keys())
            probs = list(policy[state].values())
            action = random.choices(actions, weights=probs, k=1)[0]
        else:
            action = random.choice(valid_actions)

        # Appliquer l'action
        env.step(action)
        
        # Nouvel état et score
        next_state = env.state_id()
        new_score = env.score()
        
        # Calcul reward par différence de score
        reward = new_score - previous_score
        previous_score = new_score

        # Vérifier si terminé
        done = env.is_game_over()

        episode.append((state, action, reward))
        state = next_state

    return episode



def on_policy_first_visit_mc_control(env, epsilon, num_episodes):
    Q = defaultdict(lambda: defaultdict(float))
    Returns = defaultdict(list)
    policy = defaultdict(lambda: defaultdict(float))
    gamma = 0.99

    for _ in range(num_episodes):
        episode = generate_episode(env, policy)
        G = 0

        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gamma * G + reward

            if not any(s == state and a == action for s, a, _ in episode[:i]):
                Returns[(state, action)].append(G)
                Q[state][action] = sum(Returns[(state, action)]) / len(Returns[(state, action)])

                if not policy[state]:
                    valid_actions = list(env.available_actions())
                    prob = 1.0 / len(valid_actions)
                    for a in valid_actions:
                        policy[state][a] = prob

                max_action = max(Q[state].items(), key=lambda x: x[1])[0]
                for a in policy[state]:
                    if a == max_action:
                        policy[state][a] = 1 - epsilon + (epsilon / len(policy[state]))
                    else:
                        policy[state][a] = epsilon / len(policy[state])

    return dict(policy), {k: dict(v) for k, v in Q.items()}



def off_policy_mc_control(env, num_episodes, gamma=0.99):
    """
    Implémente Off-Policy MC Control avec importance sampling.
    """
    Q = defaultdict(lambda: defaultdict(float))
    C = defaultdict(lambda: defaultdict(float))
    target_policy = {}
    behavior_policy = defaultdict(dict)
    
    for episode_num in range(num_episodes):
        episode = generate_episode(env, behavior_policy)
        G = 0.0
        W = 1.0

        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gamma * G + reward

            # Initialisation politique comportementale si besoin
            if state not in behavior_policy or not behavior_policy[state]:
                valid_actions = env.available_actions()
                prob = 1.0 / len(valid_actions)
                behavior_policy[state] = {a: prob for a in valid_actions}

            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            # Mise à jour politique cible
            best_action = max(Q[state].items(), key=lambda x: x[1])[0]
            target_policy[state] = best_action

            if action != target_policy[state]:
                break
            
            # Importance sampling ratio
            W = W / behavior_policy[state][action]
            if W == 0:
                break

    return target_policy, {k: dict(v) for k, v in Q.items()}

from collections import defaultdict
import random

from collections import defaultdict
import random

def mc_es_control(env, num_episodes, gamma=0.99):
    Q = defaultdict(lambda: defaultdict(float))
    returns = defaultdict(list)
    policy = defaultdict(lambda: defaultdict(float))

    for _ in range(num_episodes):
        # --- Exploring Start ---
        state = env.reset()
        state = env.state_id()
        valid_actions = env.available_actions()

        if valid_actions is None or len(valid_actions) == 0:
            continue

        action = random.choice(valid_actions)  # Action aléatoire initiale
        env.step(action)

        episode = []
        done = env.is_game_over()
        prev_score = env.score()
        episode.append((state, action, 0))  # Reward = 0 à l'init

        state = env.state_id()

        # --- Générer l’épisode jusqu’à la fin ---
        while not done:
            valid_actions = env.available_actions()
            if state in policy and policy[state]:
                actions = list(policy[state].keys())
                probs = list(policy[state].values())
                action = random.choices(actions, weights=probs, k=1)[0]
            else:
                action = random.choice(valid_actions)

            env.step(action)
            new_score = env.score()
            reward = new_score - prev_score
            prev_score = new_score

            episode.append((state, action, reward))
            state = env.state_id()
            done = env.is_game_over()

        # --- Mise à jour Q et politique ---
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in visited:
                visited.add((s, a))
                returns[(s, a)].append(G)
                Q[s][a] = sum(returns[(s, a)]) / len(returns[(s, a)])

                best_action = max(Q[s].items(), key=lambda x: x[1])[0]
                valid_actions = Q[s].keys()
                for act in valid_actions:
                    policy[s][act] = 1.0 if act == best_action else 0.0

    return dict(policy), {k: dict(v) for k, v in Q.items()}
