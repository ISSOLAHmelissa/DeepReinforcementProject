import random

def generate_episode(env, policy):
    episode = []
    state = env.reset()  # Now correctly returns the initial state (0)
    done = False

    while not done:
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
