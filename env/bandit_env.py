import numpy as np

# class MAB_Stochastic_M(Environment):

#     def __init__(self, means, M, noise=1.):
#         """
#         theta: d-dimensional vector (bounded) representing the hidden parameter
#         K: number of actions per round (random action vectors generated each time)
#         """
#         assert len(means) >= M, f'Got {M} subset of actions for {len(means)} arms'
#         self.theta = means
#         self.noise = noise
#         self.K = np.size(means)
#         self.M = M
    
#     def reset(self):
#         pass

#     def get_reward(self, action):
#         """ 
#         sample reward given action. Action is a list of M actions
#         """
#         assert len(action) <= self.M, f"Error in action size: {len(action)}, actions available: {self.M}"
#         returns = np.random.normal(self.theta[action], self.noise)
#         return np.max(returns)
            
#     def get_means(self):
#         return self.theta
    
class MAB_Bernoulli:
    def __init__(self, K, M, seed = 42,  corruption = 0, p_subopt = 0.1, p_opt = 0.3):
        """
        theta: d-dimensional vector (bounded) representing the hidden parameter
        K: number of actions per round (random action vectors generated each time)
        """
        # assert len(means) >= M, f'Got {M} subset of actions for {len(means)} arms'
        self.K = K
        self.M = M
        self.t = 0
        self.p_opt = p_opt
        self.p_subopt = p_subopt
        self.corruption_round = corruption
        self.reset(seed)
    
    def reset(self, seed, **kw):
        self.t = 0
        self.rng = np.random.default_rng(seed = seed)
        self.theta = np.full(self.K, self.p_subopt)
        self.best_arms = self.rng.choice(self.K, size = self.M, replace = False)
        self.theta[self.best_arms] = self.p_opt
        self.theta_corrupted = self.theta.copy()
        self.theta_corrupted[self.best_arms] = 0

    def get_reward(self, action):
        """ 
        sample reward given action. Action is a list of M actions
        """
        assert len(action) <= self.M, f"Error in action size: {len(action)}, actions available: {self.M}"
        self.t += 1
        returns = self.rng.binomial(1, p = self.theta[action])
        if self.t <= self.corruption_round:
            returns = self.rng.binomial(1, p = self.theta_corrupted[action])

        return returns
    
    def get_means(self):
        return self.theta