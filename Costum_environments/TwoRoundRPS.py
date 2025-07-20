import random

class TwoRoundRPS:
    ACTIONS = ['rock', 'paper', 'scissors']
    ACTION_TO_INDEX = {a: i for i, a in enumerate(ACTIONS)}
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.round = 1
        self.agent_history = []
        self.opponent_history = []
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.round == 1:
            return ('start',)
        elif self.round == 2:
            return ('second', self.agent_history[0])
        else:
            return ('done',)

    def available_actions(self):
        return self.ACTIONS

    def step(self, action):
        assert action in self.ACTIONS, "Action invalide"

        if self.done:
            raise Exception("Partie terminée. Appelez reset().")

        # Round 1
        if self.round == 1:
            opponent_action = random.choice(self.ACTIONS)
            self.agent_history.append(action)
            self.opponent_history.append(opponent_action)
            reward = self._get_reward(action, opponent_action)
            self.round += 1
            return self._get_state(), reward, False

        # Round 2
        elif self.round == 2:
            opponent_action = self.agent_history[0]  # Imitation
            self.agent_history.append(action)
            self.opponent_history.append(opponent_action)
            reward = self._get_reward(action, opponent_action)
            self.done = True
            return self._get_state(), reward, True

    def _get_reward(self, agent, opponent):
        if agent == opponent:
            return 0
        if (agent == 'rock' and opponent == 'scissors') or \
           (agent == 'paper' and opponent == 'rock') or \
           (agent == 'scissors' and opponent == 'paper'):
            return 1
        return -1

    def render(self):
        print(f"Round {self.round}")
        print(f"Agent history: {self.agent_history}")
        print(f"Opponent history: {self.opponent_history}")

    def is_done(self):
        return self.done

import random

class TwoRoundRPS_AgentEnv:
    ACTIONS = ['rock', 'paper', 'scissors']

    def __init__(self):
        self.reset()

    def reset(self):
        self.round = 1
        self.agent_history = []
        self.opponent_history = []
        self.done = False
        self.state = ('start',)
        return self.state

    def step(self, action):
        if self.done:
            raise Exception("Partie terminée. Appelez reset().")

        assert action in self.ACTIONS, f"Action invalide : {action}"

        # Round 1
        if self.round == 1:
            opponent_action = random.choice(self.ACTIONS)
            self.agent_history.append(action)
            self.opponent_history.append(opponent_action)
            reward = self._get_reward(action, opponent_action)
            self.round += 1
            self.state = ('second', action)
            return self.state, reward, False

        # Round 2
        elif self.round == 2:
            opponent_action = self.agent_history[0]  # imitation
            self.agent_history.append(action)
            self.opponent_history.append(opponent_action)
            reward = self._get_reward(action, opponent_action)
            self.state = ('done',)
            self.done = True
            return self.state, reward, True

    def _get_reward(self, agent, opponent):
        if agent == opponent:
            return 0
        if (agent == 'rock' and opponent == 'scissors') or \
           (agent == 'paper' and opponent == 'rock') or \
           (agent == 'scissors' and opponent == 'paper'):
            return 1
        return -1

    def available_actions(self, state=None):
        return self.ACTIONS

    def is_game_over(self):
        return self.done

    def state_id(self, state=None):
        if state is None:
            state = self.state
        return str(state)

    def score(self):
        return sum(
            self._get_reward(a, o)
            for a, o in zip(self.agent_history, self.opponent_history)
        )
    
    def is_forbidden(self, action):
        return False
    
    def num_actions(self):
        return len(self.ACTIONS)


    def display(self):
        print(f"[État actuel] : {self.state}")
        print(f"Historique Agent : {self.agent_history}")
        print(f"Historique Adversaire : {self.opponent_history}")
