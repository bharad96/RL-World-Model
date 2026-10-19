# World Models for Atari Reinforcement Learning

Final-year project by **Bharadwaj Akarapu** at **Nanyang Technological University (2021)**, supervised by **Zinovi Rabinovich**. The report, *Model-based RL in ATARI games* (SCSE20-0485), investigates how the World Models approach transfers to Atari environments.

The project adapts an existing TensorFlow implementation and explores observation preprocessing, visual representation learning and controller design. Its main findings concern the loss of small but important visual details and the difficulty of learning effective policies from the resulting representations.

## Architecture

The agent combines three components:

1. **Vision:** a convolutional variational autoencoder (VAE) compresses observation frames into a latent vector `z`.
2. **Memory:** a mixture-density LSTM (MDN-RNN) models the distribution of the next latent state from the current latent state and action.
3. **Controller:** a small linear controller with `tanh` outputs selects actions from `z` and the LSTM hidden state. CMA-ES optimizes its weights using environment rewards.

The final Boxing configuration uses 64 x 64 RGB frames, a 32-dimensional latent vector, a 256-unit LSTM and five mixture components. The modified controller produces five action scores and selects the largest. The RNN receives the selected action as a scalar, so `a_width=1` in the configuration is intentional.

Controller training and evaluation use the real Gym environment. Imagined-environment policy training is not implemented for Boxing.

## Experiments and findings

| Environment | Work completed | Main finding |
| --- | --- | --- |
| `SpaceInvaders-v0` | Observation preprocessing and representation feasibility investigation | Resizing removed small laser beams; frame-by-frame inspection also revealed flickering. Experimentation stopped before controller training. |
| `Enduro-v0` | Rollout collection, VAE training and six groups of representation experiments | Larger latent vectors improved some reconstructions qualitatively, but distant opponent cars remained unreliable. |
| `Boxing-v0` | VAE and MDN-RNN training, controller modifications and CMA-ES training/evaluation | The VAE reconstructed the players, but controller rewards remained low. The revised controller stayed around zero cumulative reward after 80 generations. |

For Enduro, **2,500 episodes of approximately 4,400 frames each** were collected and used for VAE training: about **11 million frames**. The six experiment groups covered:

- Default VAE settings with a 32-dimensional latent vector.
- Larger latent vectors of 48 and 64 dimensions.
- 100 x 100 input frames instead of 64 x 64.
- Additional convolutional encoder and transposed-convolutional decoder layers.
- Grayscale observations, including an experiment with a 256-dimensional latent vector.
- Spatially weighted reconstruction loss to emphasize distant cars.

For Boxing, the action space was reduced from 18 actions to `FIRE`, `UP`, `RIGHT`, `LEFT` and `DOWN`. A five-output argmax controller replaced threshold-based action selection, and the recurrent model was retrained. Controller experiments used MPI-based training on 64 and seven CPU cores, with 16 rollouts per candidate.

These findings are summarized from the 2021 report, which is not included in this repository. There is no demonstrated high-scoring Atari policy or measured percentage improvement in reconstruction quality. DQN results discussed in the report are contextual comparisons, not an independently reproduced baseline in this repository.

## Setup

This is a research snapshot with legacy dependencies, including Gym 0.9.4, NumPy 1.17.4 and TensorFlow Probability 0.9. Use an isolated environment compatible with those packages:

```bash
git clone https://github.com/bharad96/RL-World-Model.git
cd RL-World-Model
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Additional prerequisites are **TensorFlow compatible with the pinned TensorFlow Probability and legacy Keras APIs**, an MPI runtime providing `mpirun`, and Atari environment/ROM support for the pinned Gym release. TensorFlow itself is missing from `requirements.txt`, and the exact original Python/TensorFlow versions were not recorded. Installing current package versions is not a drop-in replacement.

The commands below reflect the checked-in scripts and configuration. Model training has not been rerun as part of this documentation update; a complete reproducible environment has not been validated.

## Training and evaluation workflow

Run from the repository root and always pass a configuration with `-c`: the parser's default points to a `doom.config` file that is no longer present. The current implementation is wired to **Boxing**, so changing only `--env_name` does not restore the historical Enduro or Space Invaders experiments.

```bash
# 1. Collect random-policy observations, actions, rewards and terminal flags.
python extract.py -c configs/boxing.config

# 2. Train the VAE on the collected frames.
python vae_train.py -c configs/boxing.config

# 3. Encode recorded observations into latent-distribution sequences.
python series.py -c configs/boxing.config

# 4. Train the MDN-RNN on those sequences and actions.
python rnn_train.py -c configs/boxing.config

# 5. Train the controller; this script launches its own MPI processes.
python train.py -c configs/boxing.config --dream_env 0
```

The Boxing configuration collects up to 250 trials, trains the VAE for one epoch and configures 40,000 RNN steps. Adjust these settings deliberately for new experiments. Raw rollout data and latent sequence datasets are not included.

Outputs live under `results/WorldModels/Boxing-v0/`:

- `record/`: compressed rollout files produced by `extract.py`.
- `tf_vae/`: VAE checkpoints and TensorBoard logs.
- `series/series.npz`: encoded observations and associated rollout data.
- `tf_rnn/`: recurrent-model checkpoints, arguments and TensorBoard logs.
- `log/`: controller parameters, training histories and periodic evaluation results.

The repository includes Boxing VAE/RNN checkpoints and archived controller logs. Running the training stages writes to the same model directories.

`train.py` performs evaluation during training; there is no separate evaluation CLI. With the checked-in configuration, it evaluates every 10 generations using seven workers and two episodes per worker (**14 evaluation episodes**). The evaluation code caps the aggregate at 100 episodes, so the comments referring to 100 episodes do not describe this configuration's actual sample count. Evaluation histories record the generation, mean, standard deviation, minimum and maximum reward.

Use `plot_training_progress.ipynb` to inspect controller histories and `Boxing-v0_viz.ipynb` or `visualizations.ipynb` to inspect model outputs. Notebook paths may need adjustment to the available artifacts. `boxing.py` is a random-agent inspection script, not a trained-policy evaluator.

## Repository layout

| Path | Purpose |
| --- | --- |
| `env.py` | Boxing observation wrapper and integration of VAE/RNN state with controller actions |
| `vae/vae.py`, `vae_train.py` | Convolutional VAE and frame-training loop |
| `rnn/rnn.py`, `rnn_train.py` | Mixture-density LSTM and sequence-training loop |
| `extract.py`, `series.py` | Rollout collection and latent-dataset preparation |
| `controller.py`, `train.py`, `es.py` | Controller, MPI training/evaluation and evolution-strategy utilities |
| `configs/` | Boxing configuration and historical Enduro configuration |
| `results/WorldModels/Boxing-v0/` | Saved models and archived run artifacts |
| `*.ipynb`, `*.svg` | Visualizations and controller training plots |

The final working tree reflects the Boxing experiment. Earlier Enduro preprocessing and VAE variants are preserved in Git history; `enduro.bash` and `configs/enduro.config` alone do not reproduce all reported experiments.

## References and attribution

- **David Ha and Jürgen Schmidhuber (2018), [World Models](https://arxiv.org/abs/1803.10122).** The architecture and overall training approach come from this work.
- An existing World Models implementation was imported in commit [`cc37d0c`](https://github.com/bharad96/RL-World-Model/tree/cc37d0cb6242042d9d8229ac5008d321ddcf9c70/ref). The current MDN-RNN and MPI trainer retain that reference implementation. Project contributions include adapting environment wrappers and data preparation, experimenting with VAE representations, modifying the discrete-action controller, and conducting and analyzing training runs.
- `es.py` credits code from OpenAI's [evolution-strategies-starter](https://github.com/openai/evolution-strategies-starter); `train.py` credits [garymcintire/mpi_util](https://github.com/garymcintire/mpi_util/) for its MPI launch helper.
- The project report also references A. Green's World Models explanation at [adgefficiency.com](https://adgefficiency.com/).
