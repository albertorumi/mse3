import numpy as np

class CascadeBandit:
    def __init__(self, E, K, T = None, seed = 42, method = 'UCB'):
        '''
        E is set of K items
        K is the number of recommended items (M)
        T time horizon
        seed seed
        method can be UCB and UCB-KL
        '''
        self.K = K
        self.E = E
        self.method = method
        assert method in set(['UCB', 'KL-UCB']), "invalid method"
        if method == 'UCB':
            self.UCBfunction = self.computeUCB
        else:
            self.UCBfunction = self.computeUCB_KL
        
        self.reset(seed=seed)
    
    def reset(self, seed):
        self.round = 0
        self.rng = np.random.default_rng(seed)
        self.counts = np.ones(self.E)
        
        self.round = 1
        self.w = self.rng.binomial(1,p = np.full(self.E, 1/2), size = self.E)

    def predict(self):
        ucbs = self.UCBfunction()
        res = ((i, u) for i,u in enumerate(ucbs))
        
        probs = sorted(res, key=lambda x: x[1], reverse = True)
        
        pulls = [p for p,_ in probs[:self.K]]
        return pulls
    
    def update(self, played, reward):
        clicked = self.K
        for i,r in enumerate(reward):
            if r == 1.0:
                clicked = i
                break
        
        self.round += 1
        used_pulls = played[:clicked + 1]
        self.counts[used_pulls] += 1
        one = np.zeros(self.w.shape)
        if clicked < self.K:
            one[played[clicked]] = 1
        self.w[used_pulls] = ((self.counts[used_pulls] - 1) * self.w[used_pulls] + one[used_pulls]) / self.counts[used_pulls]

    def computeUCB(self):
        return self.w + np.sqrt(1.5*np.log(self.round)/self.counts)
    
    def computeUCB_KL(self):
        return [self.UCB_KL(self.w[i], self.counts[i]) for i in range(self.E)]

    def UCB_KL(self, w_e, count):
        if w_e == 1 or w_e == 0:
            return w_e
        q_values = np.linspace(w_e, 0.9999999, num=10)  # Generate q values from w(e) to 1
        kl_values = count * self.bernoulli_KL(w_e, q_values)
        if self.round > 1:
            threshold = np.log(self.round) + 3 * np.log(np.log(self.round))
        else:
            threshold = 0
        valid_indices = np.where(kl_values <= threshold)[0]
        
        if len(valid_indices) > 0:
            q_star_index = valid_indices[-1]
            q_star = q_values[q_star_index]
            return q_star
        else:
            return w_e
    
    @staticmethod
    def bernoulli_KL(p, q):
        return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))
    
    def name(self):
        return f"Casc-{self.method}"