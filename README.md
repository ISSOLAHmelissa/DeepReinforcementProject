# DeepReinforcementProject

A collection of Jupyter notebooks and supporting code demonstrating classical reinforcement learning algorithms (Monte Carlo, Temporal-Difference, Dyna-Q, and Dynamic Methods) across several simple environments used for teaching and experimentation.

## Repository structure

- Costum_environments/ - custom environment implementations (empty or placeholder directories)
- DRL_algorithms/ - (planned) implementations of deep reinforcement learning algorithms
- secret_envs/ - (private/placeholder) additional environment code
- .ipynb files - multiple Jupyter notebooks implementing and demonstrating algorithms for different environments
- syllabus.pdf - course syllabus and project description

## Notebooks (high-level)

The repository contains notebooks covering the following algorithms and environments:

- monte_carlo_* and montecarlo_* - Monte Carlo policy evaluation/control examples
- TD_* and TDLearning_* - Temporal-Difference learning examples
- DynamicMethods_* - Planning / dynamic programming style examples
- DynaQ_* - Model-based Dyna-Q algorithm examples

Environments used in notebooks include:
- GridWorld
- LineWorld
- Monty Hall (Level 01 and Level 02)
- Two-Round Rock-Paper-Scissors (TwoRoundRPS)

## Requirements

Basic requirements to run the notebooks locally:

- Python 3.8+
- jupyter or jupyterlab
- numpy
- matplotlib

Optional (depending on specific notebooks):
- gym (OpenAI Gym) if any environments use its API

You can install common requirements with:

```bash
python -m pip install jupyter numpy matplotlib
# optionally: pip install gym
```

## How to run

1. Clone the repository:

```bash
git clone https://github.com/ISSOLAHmelissa/DeepReinforcementProject.git
cd DeepReinforcementProject
```

2. Install requirements (see above).
3. Start Jupyter Notebook / Lab and open any .ipynb file:

```bash
jupyter notebook
# or
jupyter lab
```

4. Run cells in the notebooks to reproduce experiments and plots.

## Notes

- Many notebooks are educational and contain explanations, visualizations and experiments for small, illustrative environments.
- Some folders (Costum_environments, secret_envs) appear as placeholders for environment code; review them before running notebooks that depend on their modules.
- If you plan to extend the project with deep RL algorithms, add dependencies (e.g., PyTorch or TensorFlow) and document them in a requirements.txt.

## License

This repository does not include a license file. If you want others to use or contribute, consider adding an open-source license (for example, MIT License).

## Contact

Repository owner: @ISSOLAHmelissa
