import random

class HumanAgentMontyHallLevel01:
    def __init__(self, env):
        self.env = env

    def play(self):
        state = self.env.reset()
        print("Bienvenue dans le jeu Monty Hall ! Trois portes : 0, 1, 2.")
        print("Une seule cache une voiture, les autres une chèvre.")
        print()

        while not self.env.is_terminal():
            print(f"État actuel : {state}")

            valid_actions = self.env.get_valid_actions()
            print(f"Actions valides : {valid_actions}")

            action = self.get_user_action(valid_actions)

            state, reward, done = self.env.step(action)

            if done:
                print(f"\n🎉 Le jeu est terminé.")
                print(f"Vous avez choisi la porte : {state[1]}")
                print(f"La porte gagnante était : {state[2]}")
                print("🎁 Gagné !" if reward == 1.0 else "🐐 Perdu, une chèvre derrière cette porte.")

    def get_user_action(self, valid_actions):
        while True:
            action = input("Votre action: ")
            try:
                if action.isdigit():
                    action = int(action)
                if action in valid_actions:
                    return action
                else:
                    print(f"Action invalide. Choisissez parmi : {valid_actions}")
            except Exception as e:
                print(f"Erreur : {e}. Réessayez.")

class HumanAgentMontyHallLevel02:
    def __init__(self, env):
        self.env = env

    def play(self):
        state = self.env.reset()
        print("\n🎮 Monty Hall Niveau 2 (Version personnalisée)")
        print("Choisissez 3 portes successives. Après chaque choix, Monty révèle une porte perdante.")
        print("À la fin, vous choisissez de garder votre 3e choix ou de switch vers l'autre porte fermée.\n")

        while not self.env.is_terminal():
            print(f"\n➡️  État actuel : {state}")
            valid_actions = self.env.get_valid_actions()
            print(f"Actions valides : {valid_actions}")
            action = self.get_user_action(valid_actions)
            state, reward, done = self.env.step(action)

            if done:
                final_choice = state[1]
                print("\n✅ Le jeu est terminé.")
                print(f"🚪 Votre porte finale : {final_choice}")
                print(f"🚪 Porte gagnante : {self.env.winning_door}")
                if reward == 1.0:
                    print("🎉 Bravo ! Vous avez gagné la voiture !")
                else:
                    print("🐐 Oups... une chèvre 😅")

    def get_user_action(self, valid_actions):
        while True:
            action = input("Votre action: ").strip().lower()
            if action.isdigit():
                action = int(action)
            if action in valid_actions:
                return action
            else:
                print(f"❌ Action invalide. Choisissez parmi : {valid_actions}")


