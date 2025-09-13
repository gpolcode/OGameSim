# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
import uuid

import gymnasium as gym
import ogame_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "my-awesome-project"
    """the wandb's project name"""
    wandb_entity: str = "gpol-none"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoint_frequency: int = 1000
    """how often a checkpoint should be saved"""
    load_checkpoint_path: str = "" # "/home/elsahr/saved_models/" + '0.04_0.98_0.94_0.3_3000_4ba05211-8211-42f8-ba26-75806c7af321'#None 
    """name of the checkpoint to load"""
    env_id: str = "ogame_env/GridWorld-v0"
    """the id of the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 300_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4000
    """the number of parallel game environments"""
    num_steps: int = 25
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.94
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.35
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    layer.weight.data = layer.weight.data.to("cuda:0")
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(math.prod(envs.single_observation_space.shape), 125)),
            nn.Tanh(),
            layer_init(nn.Linear(125, 125)),
            nn.Tanh(),
            layer_init(nn.Linear(125, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(math.prod(envs.single_observation_space.shape), 125)),
            nn.Tanh(),
            layer_init(nn.Linear(125, 125)),
            nn.Tanh(),
            layer_init(nn.Linear(125, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    torch.set_num_threads(24)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.ent_coef}_{args.gamma}_{args.gae_lambda}_{args.clip_coef}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Load checkpoint
    if args.load_checkpoint_path:
        print("Load checkpoint")
        checkpoint = torch.load(args.load_checkpoint_path)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Save a checkpoint
        if iteration % args.checkpoint_frequency == 0:
            print("Save checkpoint")
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f"/home/elsahr/saved_models/{run_name}_{str(iteration)}_{str(uuid.uuid4())}")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(
                torch.as_tensor(terminations, device=device),
                torch.as_tensor(truncations, device=device),
            )
            rewards[step] = torch.as_tensor(reward, device=device).view(-1)
            next_obs = next_obs.to(device)
            next_done = next_done.to(device)

            if "final_info" in infos:
                info = infos["final_info"]
                writer.add_scalar("charts/episodic_length_max", torch.tensor(info["episodic_length"], device=device).max(), global_step)
                writer.add_scalar("charts/episodic_length_mean", torch.tensor(info["episodic_length"], device=device).mean(), global_step)
                writer.add_scalar("charts/episodic_length_min", torch.tensor(info["episodic_length"], device=device).min(), global_step)

                writer.add_scalar("points/max", torch.tensor(info["points"], device=device).max(), global_step)
                writer.add_scalar("points/mean", torch.tensor(info["points"], device=device).mean(), global_step)
                writer.add_scalar("points/min", torch.tensor(info["points"], device=device).min(), global_step)

                writer.add_scalar("astrophysics/max",  torch.tensor(info["astrophysics"], device=device).max(), global_step)
                writer.add_scalar("astrophysics/mean", torch.tensor(info["astrophysics"], device=device).mean(), global_step)
                writer.add_scalar("astrophysics/min",  torch.tensor(info["astrophysics"], device=device).min(), global_step)

                writer.add_scalar("plasma_technology/max",  torch.tensor(info["plasma_technology"], device=device).max(), global_step)
                writer.add_scalar("plasma_technology/mean", torch.tensor(info["plasma_technology"], device=device).mean(), global_step)
                writer.add_scalar("plasma_technology/min",  torch.tensor(info["plasma_technology"], device=device).min(), global_step)

                writer.add_scalar("metal/max", torch.tensor(info["metal_max"], device=device).mean(), global_step)
                writer.add_scalar("metal/mean", torch.tensor(info["metal_mean"], device=device).mean(), global_step)
                writer.add_scalar("metal/min", torch.tensor(info["metal_min"], device=device).mean(), global_step)

                writer.add_scalar("crystal/max",  torch.tensor(info["crystal_max"], device=device).mean(), global_step)
                writer.add_scalar("crystal/mean", torch.tensor(info["crystal_mean"], device=device).mean(), global_step)
                writer.add_scalar("crystal/min",  torch.tensor(info["crystal_min"], device=device).mean(), global_step)

                writer.add_scalar("deut/max",  torch.tensor(info["deut_max"], device=device).mean(), global_step)
                writer.add_scalar("deut/mean", torch.tensor(info["deut_mean"], device=device).mean(), global_step)
                writer.add_scalar("deut/min",  torch.tensor(info["deut_min"], device=device).mean(), global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = permutation[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values, b_returns
        var_y = torch.var(y_true)
        if var_y.item() == 0:
            explained_var = float("nan")
        else:
            explained_var = (1 - torch.var(y_true - y_pred) / var_y).item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", torch.tensor(clipfracs, device=device).mean(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()