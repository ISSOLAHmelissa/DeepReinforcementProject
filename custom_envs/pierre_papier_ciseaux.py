import random

result = ((0, -1, 1),
          (1, 0, -1),
          (-1, 1, 0))

reward = 0

print("--------------PREMIER ROUND--------------")

# 0 : Pierre, 1 : Feuille, 2 : Ciseaux
first_choice_player2 = random.randint(0, 2)

first_choice_player1 = int(input("Pierre (0), Feuille (1) ou Ciseaux (2) ?"))

print(f"Choix de l'adversaire : {first_choice_player2}")
print(f"Choix du joueur : {first_choice_player1}")

if result[first_choice_player1][first_choice_player2] == 1:
    reward+=1
    print(f"Le joueur a gagné la manche ! Reward : {reward}")
elif result[first_choice_player1][first_choice_player2] == -1:
    reward-=1
    print(f"Le joueur a perdu la manche ! Reward : {reward}")
else:
    print(f"Egalité !")

print("--------------DEUXIEME ROUND--------------")

second_choice_player2 = first_choice_player1

second_choice_player1 = int(input("Pierre (0), Feuille (1) ou Ciseaux (2) ?"))

print(f"Choix de l'adversaire : {second_choice_player2}")
print(f"Choix du joueur : {second_choice_player1}")

if result[second_choice_player1][second_choice_player2] == 1:
    reward+=1
    print(f"Le joueur a gagné la manche ! Reward : {reward}")
elif result[second_choice_player1][second_choice_player2] == -1:
    reward-=1
    print(f"Le joueur a perdu la manche ! Reward : {reward}")
else:
    print(f"Egalité !")

print("--------------PARTIE TERMINEE--------------")
print(f"Reward : {reward}")