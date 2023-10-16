import numpy as np

class UCBM:
  def __init__(self, K, M, alpha):
    """
    UCB (Upper Confidence Bound) MAB algorithm

    :param K: number of arms
    :param alpha: scaling of the optimistic bonus (appears under square root)
    """
    self.alpha = alpha
    self.K = K
    self.M = M
    self.reset()

  def reset(self, **kw):
    self.t = 0
    self.avg_rewards = np.zeros(self.K)
    self.num_played = np.zeros(self.K)
    self.cumulative_reward = np.zeros(self.K)

  def predict(self):
    num_played = self.num_played + (1e-15 if np.any(self.num_played==0) else 0)
    bonuses = np.sqrt(self.alpha * np.log((self.t + 1)) / num_played)
    scores = self.avg_rewards + bonuses

    # CAREFULL: these are in inverse order...

    top_indices = np.argsort(scores)[-self.M:]

    return top_indices

  def update(self, chosen_arms, reward):
    assert len(chosen_arms) <= self.M, f"INVALID UPDATE, M: {self.M}, given len: {len(chosen_arms)}"
    self.cumulative_reward[chosen_arms] += reward
    self.num_played[chosen_arms] += 1
    self.avg_rewards[chosen_arms] = self.cumulative_reward[chosen_arms]/self.num_played[chosen_arms]

    self.t += 1

  def name(self):
    return 'UCB('+str(self.alpha)+')'