import random
from collections import defaultdict
from tqdm import tqdm

def dyna_q_secret(
    env,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1,
    n_episodes=200,
    planning_steps=10
):
    Q = defaultdict(lambda: defaultdict(float))
    model = defaultdict(dict)

    for ep in tqdm(range(n_episodes)):
        env.reset()
        state = env.state_id()

        while not env.is_game_over():
            actions = env.available_actions()
            if len(actions) == 0:  # 
                break

            # --- ε-greedy ---
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get, default=random.choice(actions))

            # --- Interagir avec l'environnement ---
            env.step(action)
            next_state = env.state_id()
            reward = env.score()
            done = env.is_game_over()

            # --- Mise à jour Q-table ---
            max_q_next = max(Q[next_state].values(), default=0)
            Q[state][action] += alpha * (reward + gamma * max_q_next - Q[state][action])

            # --- Stocker transition dans le modèle ---
            model[state][action] = (next_state, reward)

            # --- Planification ---
            for _ in range(planning_steps):
                if not model:
                    continue
                s_sim = random.choice(list(model.keys()))
                a_sim = random.choice(list(model[s_sim].keys()))
                s_p_sim, r_sim = model[s_sim][a_sim]
                max_q_sim = max(Q[s_p_sim].values(), default=0)
                Q[s_sim][a_sim] += alpha * (r_sim + gamma * max_q_sim - Q[s_sim][a_sim])

            state = next_state

    return Q
