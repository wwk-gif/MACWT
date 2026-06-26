import numpy as np

class MultiAgentController:
    """
    最小多智能体控制器（保留原先随机策略），
    支持 n_agents > 1, action_dim = 3 (w_recon, w_mean, w_tri)。
    提供 select_actions(), observe(obs), update(rewards)
    """
    def __init__(self, n_agents=3, action_dim=3):
        self.n_agents = n_agents
        self.action_dim = action_dim
        # 每个agent控制三个权重参数 [w_recon, w_mean, w_tri]
        # 初始化为合理中间值（比如 1.0）
        self.actions = np.ones((n_agents, action_dim)) * 0.5
        self.last_rewards = np.zeros(n_agents)
        self.last_obs = np.zeros(n_agents)  # per-slice scalar obs

    def select_actions(self):
        # 临时策略：每次随机调整 [-0.1, +0.1] 范围的变化（与原实现一致）
        delta = np.random.uniform(-0.3, 0.3, size=(self.n_agents, self.action_dim))
        self.actions += delta
        # 限制每个参数在合理范围内
        self.actions = np.clip(self.actions, 0.1, 2.0)
        return self.actions.copy()  # 返回 proposals，形状 (n_agents, 3)

    def observe(self, obs):
        """
        接收 per-slice 观测（长度为 n_agents 的数组），例如 per-slice recon error
        """
        try:
            obs = np.asarray(obs)
            if obs.shape[0] == self.n_agents:
                self.last_obs = obs.copy()
        except Exception:
            pass

    def update(self, rewards):
        """
        rewards: array-like shape (n_agents,) or a scalar broadcastable
        目前仍为简单记录；可替换为 RL 算法（PPO/MADDPG 等）
        """
        rewards = np.asarray(rewards)
        if rewards.size == 1:
            rewards = np.repeat(rewards, self.n_agents)
        self.last_rewards = rewards.copy()
        # 简单启发式：对获得正 reward 的 agent 轻微增加其 actions，反之减少
        lr = 0.01
        for i in range(self.n_agents):
            r = rewards[i]
            # 将 reward 的符号作为对 actions 的微调方向（这是非常简单的 heuristic）
            self.actions[i] += lr * r
        # clip again
        self.actions = np.clip(self.actions, 0.1, 2.0)
