import random
import matplotlib.pyplot as plt
import numpy as np

def generate_episode(env, policy):
    episode = []
    state = env.reset()  # Now correctly returns the initial state (0)
    done = False

    while not done:
        if state not in policy:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                print(f"[WARNING] No valid actions for state: {state}")
                break  # Exit the episode early — terminal or invalid state
            probs = [1 / len(valid_actions)] * len(valid_actions)
            action = random.choices(valid_actions, weights=probs, k=1)[0]        
        else:
           actions = list(policy[state].keys())
           probs = list(policy[state].values())
           action = random.choices(actions, weights=probs, k=1)[0]
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state

    return episode


def display_policy_console(policy):
    action_symbols = {
        0: '←',
        1: '→',
        2: '↑',
        3: '↓'
    }
    for row in range(5):
        line = ""
        for col in range(5):
            state = row * 5 + col

            if state == 0:
                # Cas Start
                if isinstance(policy.get(state), dict):
                    actions = policy[state]
                    max_val = max(actions.values())
                    best_actions = [a for a, v in actions.items() if v == max_val]
                    arrows = ''.join(action_symbols[a] for a in best_actions)
                    cell = f"S{arrows:<2}"
                elif state in policy:
                    cell = f"S{action_symbols[policy[state]]} "
                else:
                    cell = "S . "
            elif state == 4:
                cell = "<-4"
            elif state == 24:
                cell = " F  "
            elif state in policy:
                if isinstance(policy[state], dict):
                    actions = policy[state]
                    max_val = max(actions.values())
                    best_actions = [a for a, v in actions.items() if v == max_val]
                    arrows = ''.join(action_symbols[a] for a in best_actions)
                    cell = f" {arrows:<2}"
                else:
                    cell = f" {action_symbols[policy[state]]}  "
            else:
                cell = " .  "
            
            line += cell
        print(line)


def plot_monty_hall_policy(policy):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Politique apprise - Monty Hall")

    second_states = [k for k in policy.keys() if k[0] == 'second_choice']
    start_state = [k for k in policy.keys() if k[0] == 'start'][0]

    # === PARTIE 1 : Visualiser les 'second_choice' ===
    y_labels = []
    keep_probs = []
    switch_probs = []

    for state in second_states:
        label = f"{state[1]} vs {state[2]}"
        y_labels.append(label)
        keep_probs.append(policy[state].get('keep', 0))
        switch_probs.append(policy[state].get('switch', 0))

    y_pos = np.arange(len(y_labels))
    ax.barh(y_pos - 0.2, keep_probs, height=0.4, color='skyblue', label='Keep')
    ax.barh(y_pos + 0.2, switch_probs, height=0.4, color='orange', label='Switch')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Probabilité")
    ax.set_ylabel("État (premier choix vs révélé)")
    ax.legend()
    ax.grid(True)

    # === PARTIE 2 : Afficher la politique de départ ===
    fig2, ax2 = plt.subplots()
    start_probs = policy[start_state]
    doors = [0, 1, 2]
    values = [start_probs.get(d, 0) for d in doors]

    ax2.bar(doors, values, color='green')
    ax2.set_xticks(doors)
    ax2.set_xticklabels([f"Porte {d}" for d in doors])
    ax2.set_ylabel("Probabilité")
    ax2.set_title("Politique de choix initial (état 'start')")
    ax2.grid(True)

    plt.show()
    
def plot_deterministic_policy(policy):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Politique déterministe - Monty Hall")

    # === PARTIE 1 : États 'second_choice' ===
    second_states = [k for k in policy if k[0] == 'second_choice']
    labels = []
    actions = []

    for state in second_states:
        label = f"{state[1]} vs {state[2]}"
        labels.append(label)
        actions.append(policy[state])

    y_pos = np.arange(len(labels))
    colors = ['orange' if action == 'switch' else 'skyblue' for action in actions]

    ax.barh(y_pos, [1]*len(actions), color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel("Action")
    for i, action in enumerate(actions):
        ax.text(0.5, i, action, ha='center', va='center', fontsize=12, color='black', weight='bold')

    ax.grid(True)
    ax.set_title("Politique sur les états 'second_choice'")

    # === PARTIE 2 : État 'start' ===
    fig2, ax2 = plt.subplots()
    start_action = policy.get(('start', None, None))
    doors = [0, 1, 2]
    colors2 = ['green' if d == start_action else 'lightgray' for d in doors]

    ax2.bar(doors, [1]*3, color=colors2)
    ax2.set_xticks(doors)
    ax2.set_xticklabels([f"Porte {d}" for d in doors])
    ax2.set_yticks([])
    ax2.set_title("Choix initial (état 'start')")

    for i, d in enumerate(doors):
        label = "choisi" if d == start_action else ""
        ax2.text(d, 0.5, label, ha='center', va='center', fontsize=12, weight='bold')

    plt.show()

def plot_monty_hall_2_policy(policy):
    # Group by decision depth
    depth_groups = defaultdict(list)
    for state in policy:
        phase, choices, _ = state
        depth = len(choices)
        depth_groups[depth].append(state)
    
    # Plot each depth level
    fig, axes = plt.subplots(nrows=len(depth_groups), figsize=(12, 8))
    
    for depth, states in depth_groups.items():
        ax = axes[depth] if len(depth_groups) > 1 else axes
        for state in states:
            action_probs = policy[state]
            ax.bar([str(a) for a in action_probs.keys()], 
                   action_probs.values())
            ax.set_title(f"Depth {depth}: After choosing {state[1]}")
    
    plt.tight_layout()
    plt.show()
