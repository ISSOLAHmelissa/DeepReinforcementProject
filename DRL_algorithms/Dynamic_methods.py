# ======== DYNAMIC_METHODS_UNIFIED.PY ========
from typing import Any, Dict, List, Tuple
from collections import defaultdict


def iterative_policy_evaluation_sparse(
    pi: dict,
    S: list,
    A: list,
    model: dict,
    terminal_states: list,
    valid_actions_dict: dict,
    gamma: float = 0.99,
    theta: float = 1e-4
) -> dict:
    V = {s: 0.0 for s in S}
    while True:
        delta = 0.0
        for s in S:
            if s in terminal_states:
                continue
            v = V[s]
            total = 0.0
            for a, a_prob in pi.get(s, {}).items():
                for transition in model.get(s, {}).get(a, []):
                    if len(transition) == 4:
                        prob, s_p, r, _ = transition
                    else:
                        prob, s_p, r = transition
                    total += a_prob * prob * (r + gamma * V[s_p])
            V[s] = total
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_iteration_sparse(
    S: List[Any],
    A: List[Any],
    R: List[float],
    model: Dict[Any, Dict[Any, List[Tuple[float, Any, float, bool]]]],
    terminal_states: List[Any],
    valid_actions_dict: Dict[Any, List[Any]],
    gamma: float = 0.99,
    theta: float = 1e-4
) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, float]]:
    pi = {s: {a: 1.0 / len(valid_actions_dict[s]) for a in valid_actions_dict[s]} for s in S if s not in terminal_states}
    V = {s: 0.0 for s in S}

    while True:
        V = iterative_policy_evaluation_sparse(pi, S, A, model, terminal_states, valid_actions_dict, gamma, theta)
        policy_stable = True
        for s in S:
            if s in terminal_states:
                continue
            old_action = max(pi[s], key=pi[s].get)
            best_a, best_score = None, float('-inf')
            for a in valid_actions_dict[s]:
                score = 0.0
                for t in model[s][a]:
                    p, s_p, r = t[:3]
                    score += p * (r + gamma * V.get(s_p, 0.0))
                if score > best_score:
                    best_score = score
                    best_a = a
            if best_a != old_action:
                policy_stable = False
                pi[s] = {a: 1.0 if a == best_a else 0.0 for a in valid_actions_dict[s]}
        if policy_stable:
            break
    return pi, V


def value_iteration_sparse(
    S: List[Any],
    A: List[Any],
    R: List[float],
    model: Dict[Any, Dict[Any, List[Tuple[float, Any, float, bool]]]],
    terminal_states: List[Any],
    valid_actions_dict: Dict[Any, List[Any]],
    gamma: float = 0.99,
    theta: float = 1e-4
) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, float]]:
    V = {s: 0.0 for s in S}
    while True:
        delta = 0.0
        for s in S:
            if s in terminal_states:
                continue
            v = V[s]
            best_score = float('-inf')
            for a in valid_actions_dict[s]:
                score = 0.0
                for t in model[s][a]:
                    p, s_p, r = t[:3]
                    score += p * (r + gamma * V.get(s_p, 0.0))
                best_score = max(best_score, score)
            V[s] = best_score
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    pi = {}
    for s in S:
        if s in terminal_states:
            continue
        best_a, best_score = None, float('-inf')
        for a in valid_actions_dict[s]:
            score = 0.0
            for t in model[s][a]:
                p, s_p, r = t[:3]
                score += p * (r + gamma * V.get(s_p, 0.0))
            if score > best_score:
                best_score = score
                best_a = a
        pi[s] = {a: 1.0 if a == best_a else 0.0 for a in valid_actions_dict[s]}
    return pi, V
