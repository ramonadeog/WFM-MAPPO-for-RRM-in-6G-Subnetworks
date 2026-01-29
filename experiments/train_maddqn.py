# wfmmarl_simulator/experiments/train_maddqn.py
import yaml
import numpy as np
import torch

from environment.subnetwork_env import SubnetworkEnv, EnvConfig
from marl.baselines.maddqn.maddqn_agent import MAD_DQN

def main():
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    env_cfg = EnvConfig(**cfg["environment"])
    env = SubnetworkEnv(env_cfg)

    device = cfg["device"]
    agent = MAD_DQN(
        env,
        buffer_capacity=cfg["experiment"]["maddqn_buffer"],
        batch_size=cfg["experiment"]["maddqn_batch"],
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        tau=0.01,
        device=device,
    )

    episodes = cfg["experiment"]["training_steps"]
    T = cfg["experiment"]["rollout_length"]  # reuse rollout length as episode horizon

    for ep in range(episodes):
        obs = env.reset()
        sum_rate = 0.0

        for t in range(T):
            actions = agent.act(obs)
            next_obs, rewards, _done, info = env.step(actions)
            # Store transition for all agents
            agent.buffer.add(obs, actions, rewards, next_obs, np.zeros(env.cfg.N, dtype=np.float32))
            # One gradient step per env step
            agent.train_step()
            sum_rate += info["rates_bps"].sum()
            obs = next_obs

        # Soft update after each episode
        agent.update_targets()

        if ep % 20 == 0:
            print(f"[Episode {ep}] Sum-rate={sum_rate/1e6:.2f} Mbps, epsilon={agent.epsilon:.3f}")

    # Save learned Q-nets
    torch.save(agent.qnets.state_dict(), "maddqn_agents.pt")
    print("MAD-DQN training complete!")

if __name__ == "__main__":
    main()