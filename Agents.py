import numpy as np

class MultiAgentController:
    """
    稳定 + 局部贪心搜索版（最接近真实最优步长）
    """
    def __init__(self, n_agents=3, action_dim=3):
        self.n_agents = n_agents
        self.action_dim = action_dim

        self.actions = np.ones((n_agents, action_dim)) * 0.5
        self.last_rewards = np.zeros(n_agents)

        # 小步长基础探索 0.13
        self.step_size = np.ones(n_agents) * 0.13

        # 试探步长（关键！）
        self.probe_eps = 0.05

        # 平滑奖励
        self.smooth_rewards = np.zeros(n_agents)
        self.alpha = 0.4

    def select_actions(self):
        """
        返回 actions，但不更新。
        更新动作在 update() 中完成。
        """
        return self.actions.copy()

    def update(self, rewards, evaluate_fn=None):
        """
        rewards: 当前 actions 的 reward
        evaluate_fn: 需要传入 evaluate(action_matrix) → reward 向量
        """
        rewards = np.asarray(rewards)

        if rewards.size == 1:
            rewards = np.repeat(rewards, self.n_agents)

        # 更新平滑奖励
        for i in range(self.n_agents):
            self.smooth_rewards[i] = (
                self.alpha * self.smooth_rewards[i] +
                (1 - self.alpha) * rewards[i]
            )

        # ========= 核心：试探与贪心 ==========
        new_actions = np.zeros_like(self.actions)

        for i in range(self.n_agents):

            base = self.actions[i]

            # 小步长随机探索
            explore = base + np.random.uniform(
                -self.step_size[i], self.step_size[i], size=self.action_dim
            )

            # 局部试探动作（关键）
            probe_plus = base + self.probe_eps
            probe_minus = base - self.probe_eps

            # 裁剪范围
            explore = np.clip(explore, 0.1, 2.0)
            probe_plus = np.clip(probe_plus, 0.1, 2.0)
            probe_minus = np.clip(probe_minus, 0.1, 2.0)

            # 需要 evaluate_fn 支持单个切片的 reward 计算
            if evaluate_fn is not None:
                r0 = self.smooth_rewards[i]
                rp = evaluate_fn(i, probe_plus)    # 正方向 reward
                rm = evaluate_fn(i, probe_minus)   # 反方向 reward
                re = evaluate_fn(i, explore)       # 随机方向 reward

                # 选择最好的动作
                best = np.argmax([r0, rp, rm, re])
                if best == 1:
                    chosen = probe_plus
                elif best == 2:
                    chosen = probe_minus
                elif best == 3:
                    chosen = explore
                else:
                    chosen = base
            else:
                chosen = explore  # 没提供 evaluate_fn 则退化成你的旧逻辑

            new_actions[i] = chosen

            # 步长更新（更小的变化）
            if rewards[i] > self.last_rewards[i]:
                self.step_size[i] *= 1.15
            else:
                self.step_size[i] *= 0.85

            self.step_size[i] = np.clip(self.step_size[i], 0.02, 0.3)

        # 更新
        self.actions = np.clip(new_actions, 0.1, 2.0)
        self.last_rewards = rewards.copy()
