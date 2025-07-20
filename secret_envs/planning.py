import numpy as np
import random

def dyna_q(env, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=10, episodes=1000):
    num_states = env.num_states()
    num_actions = env.num_actions()

    Q = np.zeros((num_states, num_actions))
    Model = dict()
    visited_states = set()

    for ep in range(episodes):
        env.reset()
        s = env.state_id()
        visited_states.add(s)

        while not env.is_game_over():
            available = env.available_actions()
            if len(available) == 0:
                break

            # epsilon-greedy
            if random.random() < epsilon:
                a = random.choice(available)
            else:
                a = max(available, key=lambda x: Q[s, x])

            env.step(a)
            s_next = env.state_id()
            r = env.score()

            # Q-learning update
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])

            # Store in model
            Model[(s, a)] = (s_next, r)

            # Planning
            for _ in range(planning_steps):
                s_pl, a_pl = random.choice(list(Model.keys()))
                s_p, r_p = Model[(s_pl, a_pl)]
                Q[s_pl, a_pl] += alpha * (r_p + gamma * np.max(Q[s_p]) - Q[s_pl, a_pl])

            s = s_next
            visited_states.add(s)

    # Build policy for visited states only
    policy = {}
    for s in visited_states:
        available_actions = [a for a in range(num_actions) if Q[s, a] != 0 or (s, a) in Model]
        if available_actions:
            best_a = max(available_actions, key=lambda a: Q[s, a])
            policy[s] = best_a

    return policy, Q
