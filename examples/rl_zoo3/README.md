To Train panda reacher:

```bash
python -m rl_zoo3.train \
--algo tqc \
--env PandaReach-v3 \
--conf-file hyperparams/tqc.yml \
--tensorboard-log ./logs \
--save-freq 100_000 \
--eval-freq 10_000 \
--progress
```
breakdown of the command:
- `python -m rl_zoo3.train` is the command to train an agent
- `--algo tqc` specifies the algorithm to use, in this case Twin Q-Value Critic
- `--env PandaReach-v3` specifies the environment to train the agent in
- `--conf-file hyperparams/tqc.yml` specifies the hyperparameters to use for training, if not specified, default hyperparameters from the rl_zoo3/hyperparams directory are used
- `--tensorboard-log ./logs` specifies the directory to save tensorboard logs
- `--progress` is an optional argument that enables the progress bar

to check tensorboard logs:

```bash
tensorboard --logdir ./logs/PandaReach-v3/TQC_1 --host 0.0.0.0
```

breakdown of the command:
- `tensorboard --logdir ./logs/PandaReach-v3/TQC_1` specifies the directory to load tensorboard logs from
- `--host 0.0.0.0` specifies the host to run tensorboard on, in this case, it is run on `0.0.0.0` so that it can be accessed from a remote machine

to train kuka reacher:

```bash
python -m rl_zoo3.train --algo tqc --env RaccoonKr300R2500UltraReach-v1 --conf-file hyperparams/tqc.yml --tensorboard-log ./logs --save-freq 30_000 --eval-freq 5_000 --progress
```



To run a trained agent:

```bash
python -m rl_zoo3.enjoy --algo ppo --env CartPole-v1 --folder ./trained_models/ppo/CartPole-v1
```

To run a trained agent with a specific seed:

```bash
python enjoy.py --algo <ALGO> --env <ENV> --folder <TRAIN_AGENT_FOLDER> --env-kwargs render_mode:human
```

python -m rl_zoo3.record_video --algo tqc --env PandaReach-v3 --exp-id 1 --folder ./logs -n 100 --env-kwargs render_mode:rgb_array
Raccoon Gym
Train Reach

python -m rl_zoo3.train \
--algo tqc \
--env RaccoonKr300R2500UltraReach-v1 \
--conf-file hyperparams/tqc.yml \
--tensorboard-log ./logs \
--save-freq 5_000 \
--eval-freq 1_000 \
--progress

Render Reach

python -m rl_zoo3.record_video --algo tqc --env RaccoonKr300R2500UltraReach-v1 --exp-id 1 --folder ./logs -n 500 --env-kwargs render_mode:rgb_array
