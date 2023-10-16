import numpy as np

class Bernoulli_adv_phase:
    def __init__(self, K, M, n_best_arms, delta, seed = 42):
        """
        Reward based initialization
        """
        self.rng = np.random.default_rng(seed=seed)
        self.K = K
        self.M = M
        self.n_best_arms = n_best_arms

        optimal_arm_idx = self.rng.choice(self.K, self.n_best_arms, replace=False)
        self.optimal_arm_idx = optimal_arm_idx
        self.theta_1 = np.full(K, 1/K)
        self.theta_1[optimal_arm_idx] = 1/K + delta
        
        self.theta_2 = np.full(K, 1 - delta)
        self.theta_2[optimal_arm_idx] = 1
        self.reset()

    def reset(self, seed):
        self.rng = np.random.default_rng(seed=seed)
        self.phase = 1
        self.t = 0
        self.threshold = 1

    def get_reward(self, action):
        """ 
        sample reward given action. Action is a list of M actions
        """
        assert len(action) <= self.M, f"Error in action size: {len(action)}, actions available: {self.M}"
        self.t += 1
        if self.t >= self.threshold:
            self.phase = -self.phase
            self.threshold = self.threshold * 1.6

        theta = self.theta_1 if self.phase == 1 else self.theta_2
        returns = self.rng.binomial(1, theta)
        return returns[action]

    def get_means(self):
        theta = self.theta_1 if self.phase == 1 else self.theta_2
        return theta
    

class Bernoulli_adv_dependent:
    def __init__(self, K, M, T, seed = 42):
        """
        Reward based initialization
        """
        self.M = M
        self.K = K
        self.T = T
        self.reset(seed)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed=seed)
        self.t = 0
        self.threshold = 100
        sigma_sq = 1 / (192 + 96*np.log(self.T))
        epsilon = np.sqrt(self.K * self.M/ (8 * self.T)) * np.sqrt(sigma_sq)

        self.best_arms = self.rng.choice(self.K, self.M, replace = False) 
        means = np.full(self.K, 1/2)
        means[self.best_arms] = 1/2 + epsilon
        self.theta_vector = self.rng.normal(loc = means, scale=sigma_sq)

    def get_reward(self, action):
        """ 
        sample reward given action. Action is a list of M actions
        """
        assert len(action) <= self.M, f"Error in action size: {len(action)}, actions available: {self.M}"
        theta = self.theta_vector
        returns = self.rng.binomial(1, theta)
        return returns[action]

class Jumping_env:
    def __init__(self, K, M, T, p_opt, p_subopt, seed = 42):
        """
        Reward based initialization
        """
        assert len(p_opt) == 2, "format error"
        self.K = K
        self.M = M
        self.p_opt = p_opt
        self.p_subopt = p_subopt
        self.reset()


    def reset(self, seed = 42):
        self.rng = np.random.default_rng(seed=seed)
        self.t = 0
        self.threshold = 100
        self.best_arms = self.rng.choice(self.K, self.M, replace = False) 
        self.theta_vector = np.full(self.K, self.p_subopt)
        self.theta_vector[self.best_arms] = self.p_opt[0]

    def get_reward(self, action):
        """ 
        sample reward given action. Action is a list of M actions
        """
        assert len(action) <= self.M, f"Error in action size: {len(action)}, actions available: {self.M}"
        theta = self.theta_vector.copy()
        rand_best = self.rng.choice(self.best_arms)
        theta[rand_best] = self.p_opt[1]
        returns = self.rng.binomial(1, theta)
        return returns[action]

    # def get_means(self):
    #     theta = self.theta_1 if self.phase == 1 else self.theta_2
    #     return theta
# class Bernoulli_adv_dependent:
#     def __init__(self, K, M, p_best = (0.5, 0.1), p_subopt = 0.2, seed = 42, corruption = 0):
#         """
#         Reward based initialization
#         """
#         self.reset()
#         self.M = M
        
#         self.best_arms = self.rng.choice(K, M, replace = False) 
#         self.theta_vector = {
#             phase : np.array([p_subopt for _ in range(K)]) for phase in range(M)
#         }
#         for i in range(M):
#             self.theta_vector[i][self.best_arms[i]] = p_best[0]
#             self.theta_vector[i][self.best_arms] = p_best[1]

#         self.corruption = corruption

#     def reset(self, seed = 42):
#         self.rng = np.random.default_rng(seed=seed)
#         self.phase = 0
#         self.t = 0
#         self.threshold = 100

#     def get_reward(self, action):
#         """ 
#         sample reward given action. Action is a list of M actions
#         """
#         assert len(action) <= self.M, f"Error in action size: {len(action)}, actions available: {self.M}"
#         self.t += 1
#         if self.t >= self.threshold:
#             self.phase = (self.phase + 1) % self.M
#             self.threshold += 10

#         theta = self.theta_vector[self.phase]
#         returns = self.rng.binomial(1, theta)
#         if self.corruption > 0:
#             returns[self.best_arms] = 0
#             for i, r in enumerate(returns):
#                 if r == 1:
#                     returns[i] = 0
#             self.corruption -= 1
#         return returns[action]

#     # def get_means(self):
#     #     theta = self.theta_1 if self.phase == 1 else self.theta_2
#     #     return theta