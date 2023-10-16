import numpy as np

class EXPM:
    def __init__(self, K, M, reward_function = max, T = None, eta = 0.01, seed = 42):
        """
        Initialize
        """
        self.K = K
        self.M = M
        if T != None:
            R = 2 * np.sqrt(2 * np.log(K) * M * T * (K + M - 1))
            eta = np.log(K) / R
            # eta = np.sqrt(4 * np.log(K+1) / (M*T*(self.K+M)))
            print("MSE3 theoretically initialized, eta: ", eta)
        self.eta = eta
        self.reward_function = reward_function
        self.reset(seed = seed)
    
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.p = np.array([1/(self.K) for _ in range(self.K)])

    def predict(self):
        '''
        Sample M arms with replacement from sampling distribution
        '''
        res = self.rng.choice(self.K, self.M, p = self.p)
        return res
    
    def update(self, played, reward, verbose = False):
        '''
        Update distribution based on the reward received.
        inputs: 
            set : set of arms played
            int : binary reward
        '''
        assert len(played) <= self.M, "FORMAT ERROR"
        # if reward >= 1:
        #     # Don't update
        #     if verbose:
        #         print("END UPD: ", self.p)
        #     return
        
        # l_t_ex = np.ones(self.p.shape)
        # for m in played:
        #     l_t_ex[m] = np.exp(-self.eta * (1/self.p[m]))
        #self.p[m] = self.p[m] * np.exp(-self.eta * (1/self.p[m]))
        #p = self.p * l_t_ex
        temp = self.p
        rew = self.reward_function(reward)
        factor = np.clip(self.eta * rew/self.p[played], None, 1e+02)
        self.p[played] = self.p[played] * np.exp(factor)
        fac = sum(self.p)
        self.p = self.p / fac
        if verbose:
            print("END UPD: ", self.p)
    
    def name(self):
        return "MSE3"