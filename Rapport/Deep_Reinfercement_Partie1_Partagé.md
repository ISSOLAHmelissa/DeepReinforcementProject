
# Présentation des algorithmes implémentés
## 1. Programmation dynamique

### Policy Evaluation

L’algorithme d’évaluation itérative de politique permet d’estimer la **valeur d’un état** `V(s)` sous une politique donnée `π`.  
Cette valeur représente la récompense moyenne cumulée que l’agent peut espérer obtenir en partant de cet état et en suivant la politique.

L’algorithme met à jour les valeurs `V(s)` de manière successive à l’aide de la **relation de Bellman** en prenant en compte toutes les actions possibles `a`, pondérées par leur probabilité `π(a|s)`, ainsi que tous les états suivants `s'` et les récompenses possibles `r`, pondérés par les probabilités de transition `p(s', r | s, a)`.

Le processus se poursuit jusqu’à **convergence**, c’est-à-dire tant que la variation maximale entre deux itérations successives reste supérieure à un seuil prédéfini.

Les **états terminaux** sont traités comme des points fixes : leur valeur reste constante à zéro, puisqu’aucune récompense future n’est attendue une fois ces états atteints.

Cette méthode est applicable à tout environnement défini par :
- un ensemble d’états `S`
- un ensemble d’actions `A`
- une fonction de transition probabiliste `p`
- et une politique `π`.
###  Expérimentation : Policy Evaluation sur GridWorld

**Environnement** : GridWorld  
**Paramètres** :
- `θ = 0.0001`
- `γ = 0.99`
- États terminaux : reward = -3 (haut droite), reward = +1 (bas droite)

####  Analyse de la politique : Always Up

![[AlwaysUp_EvaluationPolicy.png]]

- Les valeurs des états sont très faibles, proches de 0.
- L’agent remonte dans la grille sans jamais atteindre les états terminaux positifs.
- Dans certains cas, il atteint l’état terminal négatif en haut à droite (récompense -3).
- Ce comportement est cohérent : monter ne mène jamais à la récompense +1.

#### Analyse de la politique : Alawys down
![[AlwaysDown_EvaluationPolicy.png]]

- Valeurs élevées (~1.0) près du terminal à +1 (état 24).
- Politique très efficace pour atteindre le terminal positif.
- Preuve que cette stratégie maximise les récompenses dans ce cas précis.

#### Analyse de la politique : Alawys Right
![[AlawysRight_EvaluationPolkicy.png]]

- Fortes valeurs négatives au début de la grille (proximité du terminal -3).
- Valeurs proches de +1 dans la dernière ligne : trajectoire efficace vers le terminal positif.
- Politique qui commence mal mais s’oriente bien vers la fin.

#### Analyse de la politique : Alawys Left
![[AlwaysLeft_EvaluationPolicy.png]]

- Valeurs faibles et relativement constantes.
- L’agent semble rester “bloqué” dans des zones sans terminal.
- Politique peu efficace : ne mène ni vers +1 ni vers -3.

#### Analyse de la politique : Uniforme aléatoire
![[UniformeAleatoire.png]]
- Valeurs majoritairement négatives.
- Indique que l’agent subit souvent la pénalité du terminal négatif.
- Quelques valeurs positives apparaissent quand la politique permet parfois d’atteindre +1.
- Comportement “aléatoire” donc résultats médiocres mais variables.

#### Conclusion Générale

- Les résultats sont cohérents avec les politiques définies.
- `Always Down` et `Always Right` sont les meilleures politiques pour atteindre l’état +1.
- `Always Up` conduit à l’état terminal -3, donc produit des valeurs faibles/négatives.
- `Always Left` garde l’agent dans une boucle improductive.
- La politique uniforme montre les limites d’un comportement sans objectif.


### Policy Iteration 

L’algorithme **Policy Iteration** repose sur deux étapes principales, exécutées de manière itérative :  
1. l’évaluation de la politique actuelle,  
2. son amélioration en sélectionnant les meilleures actions.


#### Policy Evaluation

On commence avec une **politique initiale** `π` choisie aléatoirement (dans le code : `pi = np.array([np.random.choice(A) for s in S])`).  
L’objectif est de calculer les valeurs d’états `V(s)` que l’agent obtiendrait en **suivant cette politique**, c’est-à-dire :
```math
V(s) = \sum_{s', r} p(s, \pi(s), s', r) \cdot \left[ r + \gamma V(s') \right]
```

Cette étape est implémentée dans une boucle `while`, où l’on met à jour chaque `V[s]` jusqu’à ce que la variation maximale `delta` entre deux itérations devienne inférieure à un seuil `θ`.

dans le code Les **états terminaux** `T` sont initialisés avec une valeur nulle (`V[T] = 0.0`) car l’épisode se termine dès qu’on les atteint.

#### Policy Improvement
À partir des valeurs estimées, on met à jour la politique en choisissant, pour chaque état `s`, l’action `a` qui maximise l’espérance de retour,Si la politique ne change plus, l’algorithme s’arrête.

###  Expérimentation : Policy Iteration sur GridWorld
#### Paramètres utilisés pour le test

- **Environnement** : GridWorld 5x5
- **Récompenses** :
    
    - État terminal en haut à droite (état 4) : **-3.0**
    
    - État terminal en bas à droite (état 24) : **+1.0**

- **Facteur d’actualisation** : γ=0.99\gamma = 0.99
- **Seuil de convergence** : θ=10−4\theta = 10^{-4}

![[policyiteration_gridworld.png]]
```python
[1 1 1 3 0
 1 1 1 1 3
 1 1 1 1 3
 1 1 1 1 3
 1 1 1 1 0]
 ```
 
#### Politique optimale trouvée 

```
[→ → → ↓ x
 → → → → ↓
 → → → → ↓
 → → → → ↓
 → → → → x]
```

* L’agent se déplace vers la **droite**, puis descend verticalement vers la sortie à récompense **+1** en bas à droite.

* Il **évite l’état terminal à -3** (en haut à droite), ce qui est un comportement intelligent et optimal.

* La dernière action `0` à l’état 24 est arbitraire, car c’est un **état terminal**.

#### Analyse des valeurs d’état (`V_opt`)

```python

[0.93  0.94  0.95  0.96  0.00

 0.94  0.95  0.96  0.97  0.98

 0.95  0.96  0.97  0.98  0.99

 0.96  0.97  0.98  0.99  1.00

 0.97  0.98  0.99  1.00  0.00]

```

   Ces valeurs montrent que :

* Plus on se rapproche de **l’état terminal positif (24)**, plus la valeur est haute (≈1).

* L’état terminal négatif (état 4) a une valeur de 0, car aucune récompense n’est obtenue après son entrée.

* Les états menant vers l’état 4 sont **évités naturellement**, car leurs valeurs sont inférieures à ceux qui mènent à l’état 24.
*  **suivre une stratégie stable** qui maximise la récompense cumulée.

#### Conclusion

> La politique optimale trouvée par `Policy Iteration` dirige l’agent vers l’état terminal situé en bas à droite (état 24), qui donne une récompense positive de +1.

> En revanche, elle évite l’état terminal en haut à droite (état 4), qui donne une récompense négative de -3.

> Cela montre que l’algorithme a correctement appris à maximiser les récompenses à long terme, en exploitant la structure de l’environnement.

###  Value Iteration 

**Value Iteration** est un algorithme plus rapide que Policy Iteration car il **combine l’évaluation et l’amélioration dans une seule étape**.  
À chaque itération, il met à jour la valeur de chaque état `V(s)` en utilisant directement la **meilleure action possible**, selon la **formule de Bellman optimalité** :

```math
V(s) \leftarrow \max_a \sum_{s', r} p(s, a, s', r) \cdot [r + \gamma V(s')]
```
L’algorithme s’arrête lorsque les valeurs ne changent presque plus.  On en déduit ensuite la politique optimale en choisissant, pour chaque état, **l’action qui maximise cette valeur**.

###  Expérimentation : Value Iteration sur GridWorld

![[Valueiteration_gridworld.png]]
- La politique optimale issue de Value Iteration est exactement la même que celle obtenue avec Policy Iteration, ce qui confirme la cohérence des deux méthodes.
    
- Les valeurs les plus élevées (~1)apparaissent près de l’état terminal positif en bas à droite.
    
- Les valeurs diminuent progressivement en s’éloignant de cet objectif, ce qui est cohérent avec un facteur de discount `γ = 0.99`.
    
- Les actions `1` (droite →) et `3` (bas ↓) dominent, ce qui indique que l’agent cherche à atteindre la sortie positive.
    
- L’état terminal négatif (état 4, en haut à droite) est **évité** par la politique, montrant que l’agent apprend à **minimiser les punitions**.

#### Conclusion

L’algorithme **Value Iteration** parvient à apprendre une politique optimale équivalente à celle de **Policy Iteration**, mais avec une convergence plus directe.  
La fonction de valeur obtenue reflète parfaitement la dynamique de l’environnement : des valeurs croissantes vers l’état terminal positif et des choix stratégiques pour éviter les états défavorables.

## 2. Méthodes Monte Carlo
### Monte Carlo Exploring Starts ES

Monte Carlo ES repose sur l’idée de **démarrages aléatoires** (exploring starts), permettant d’assurer une **exploration complète de toutes les paires état-action** en débutant chaque épisode dans un état et une action choisis aléatoirement.

Après chaque épisode **complet**, on met à jour la fonction de valeur d’action Q(s,a)en **moyennant les récompenses cumulées observées** pour cette paire.

Ensuite, on **améliore la politique** en choisissant, pour chaque état, l’action qui maximise Q(s,a)
, Cette méthode est **garantie de converger vers la politique optimale**, **à condition que toutes les paires état-action soient explorées**.

Elle s’applique à tout environnement **épisodique** (avec une fin d’épisode bien définie), **où les récompenses peuvent être accumulées sur des trajectoires complètes**, et **où l’on peut forcer un démarrage aléatoire**.
### Expérimentation : Monte Carlo ES sur GridWorld
#### Paramètres d’entrée utilisés :

- `episodes_count = 100000`
- `gamma = 0.99`

![[montecarloes_gridworld.png]]

#### Politique optimale estimée

La politique optimale estimée, qui associe à chaque état l’action offrant la meilleure espérance de retour cumulatif, est la suivante :

```
[3 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 0]
```

🔹 L’agent suit une trajectoire vers le **terminal positif** (état 24) situé en bas à droite.  
🔹 Il **évite l’état 4** (haut droite) qui mène à une **récompense négative (-3)**.  
🔹 Le dernier état étant terminal, son action (`0`) est arbitraire.

Cette politique indique une préférence pour les mouvements vers le bas (action 3) et vers la gauche (action 0) dans la plupart des états initiaux, tandis qu’en approchant des états terminaux, les actions s’orientent vers la droite (action 1) ou restent statiques dans l’état terminal.

#### Valeurs Q(s, a) estimées

Le tableau des valeurs Q donne, pour chaque couple état-action, une estimation de la qualité à long terme de choisir cette action. Les observations clés sont :

- Les valeurs négatives ou proches de zéro pour certains états reflètent la pénalité de -3 associée au terminal en haut à droite.
    
- Les valeurs proches de 1 correspondent aux actions qui conduisent vers l’état terminal positif en bas à droite.
    
- Ces estimations corroborent la politique optimale, confirmant que l’algorithme a correctement identifié les actions maximisant le retour espéré.

#### Pourquoi Monte Carlo ES est tres lent ?

1. **Basé sur des episodes complets** : Contrairement aux méthodes de programmation dynamique ( comme value iteration), Monte carlo ne met a jour ses valeurs qu'a la fin d'un episode complet
2. **Exploration aléatoire** : l'algorithme force l'exploration de toutes les paires ( état, action) en démarrant aléatoirement dans chaque episode, mais certaines trajectoires mènent à des états peu informatifs ou loin des terminaux, ce qui ralentit l’apprentissage utile.
3. **Apprentissage par moyenne lente** : L’algorithme met à jour les valeurs Q(s,a)Q(s, a)Q(s,a) en prenant une moyenne des retours observés ; mais plus une paire (s,a)(s, a)(s,a) est visitée, plus chaque nouvelle mise à jour a peu d’impact, ce qui rend la convergence très lente et explique le temps d’exécution élevé.

#### Conclusion
>L’algorithme **Monte Carlo ES** permet d’estimer une politique optimale dans un environnement inconnu, en s’appuyant sur des trajectoires complètes avec exploration aléatoire.  
>Le comportement appris est **cohérent avec les récompenses** : aller vers l’état à +1, éviter l’état à -3.  
>Ce résultat montre que **même sans modèle** (pas besoin des probabilités de transition), on peut converger vers une bonne politique grâce à l’exploration et à la moyenne des retours.  
>Cependant, cette méthode utilise une moyenne incrémentale pour mettre à jour les valeurs Q, ce qui rend chaque mise à jour moins influente à mesure que le nombre de visites augmente.  
>Cela explique la **convergence lente** et la **durée d’exécution importante** observées, un comportement typique des méthodes Monte Carlo nécessitant beaucoup d’épisodes pour stabiliser leurs estimations.

## 3. Apprentissage par différence temporelle (TD Learning)
### Q-Learning

L’algorithme **Q-learning** est une méthode d’apprentissage par renforcement **off-policy**, où l’agent apprend une fonction de valeur d’action Q(s,a)Q(s,a)Q(s,a) qui estime la récompense cumulée maximale qu’il peut espérer en prenant l’action aaa dans l’état sss puis en suivant la meilleure politique possible.

À chaque épisode, l’agent explore l’environnement en choisissant parfois des actions aléatoires (exploration) et d’autres fois en exploitant ses connaissances actuelles (exploitation). Après chaque action, il met à jour la table QQQ en se basant sur la récompense reçue et la meilleure valeur QQQ possible de l’état suivant.

Les paramètres clés sont :

- le taux d’apprentissage α\alphaα, qui contrôle la vitesse de mise à jour des valeurs,
    
- le facteur d’actualisation γ\gammaγ, qui valorise les récompenses futures,
    
- et le taux d’exploration ϵ\epsilonϵ, qui équilibre exploration et exploitation.

### Expérimentation : Q-learning sur GridWorld
#### Paramètres de l’algorithme Q-learning

- **Nombre d’épisodes d’apprentissage (episodes_count)** : 100 000  
    Nombre total d’épisodes pendant lesquels l’agent explore et apprend.
    
- **Taux d’exploration (ε - epsilon)** : 0,1  
    Probabilité de choisir une action aléatoire pour favoriser l’exploration (10 % des actions sont choisies aléatoirement).
    
- **Facteur d’actualisation (γ - gamma)** : 0,99  
    Importance accordée aux récompenses futures par rapport aux récompenses immédiates.
    
- **Taux d’apprentissage (α - alpha)** : 0,1  
    Vitesse à laquelle les nouvelles informations (récompenses et valeurs futures) mettent à jour les valeurs Q existantes.
    ![[Q-Learning_Gridworld.png]]
    
```
[→ → → ↓  x
 → → → ↓  ↓
 → → → ↓  ↓
 → → → ↓  ↓
 → → → →  x]
```

- La politique optimale guide l’agent vers le coin bas-droit de la grille (récompense +1), tout en évitant l’état terminal en haut à droite (récompense négative -3).
    
- Les valeurs Q reflètent bien cette stratégie : valeurs plus élevées vers les états proches du terminal positif, valeurs basses ou négatives près du terminal négatif.
    
- La convergence est rapide : ici, 100 000 épisodes suffisent pour apprendre une politique stable et cohérente.
    
- Le taux d’exploration ε = 0,1 permet un bon équilibre entre exploration et exploitation pendant l’apprentissage.
    
- La fonction Q capture efficacement la qualité des actions dans chaque état, facilitant l’amélioration progressive de la politique.


### Conclusion

Le Q-learning permet à l’agent d’apprendre une politique optimale en interaction directe avec l’environnement, sans connaissance préalable des probabilités de transition, ce qui le rend adapté aux environnements complexes ou partiellement connus.  
Les résultats confirment que l’agent a correctement appris à naviguer dans la grille vers les états souhaités en maximisant les récompenses cumulées.

Bien sûr, voici un exemple de texte que tu peux mettre dans ton rapport pour comparer tous ces algorithmes, leurs forces, faiblesses, et particularités sur ton problème GridWorld :

