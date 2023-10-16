import scipy.special
import numpy as np
import random

def _dumb_sampling(probs, M, **kw):
    return np.random.choice(len(probs), size= M, p = probs)


def _metropolis_hastings(probs, M, iterations=1000, **kw):
    K = len(probs)
    # Initialize a random subset of size M
    S = random.sample(range(K), M)
    for _ in range(iterations):
        # Propose a new subset by swapping an element
        swap_out = random.choice(S)
        swap_in = random.choice([i for i in range(K) if i not in S])
        new_S = S.copy()
        new_S.remove(swap_out)
        new_S.append(swap_in)
        # Compute acceptance ratio
        r = (probs[swap_in] / probs[swap_out]) ** M
        # Accept new_S with probability min(1, r)
        if random.random() < r:
            S = new_S
    return S

class CombBand:
    def __init__(self, K, M, sampling_function = _dumb_sampling, T = None, eta = 0.01, gamma = 0.01, seed = 42):
        self.K = K
        self.M = M
        if T is not None:
            self.N = int(scipy.special.comb(K,M))
            B = np.sqrt(M)
            self.T = T
            eta = 1 / B * np.sqrt(np.log(self.N) / (T * (M/(B**2) + 2)))
            print("Comband theoretically initialized: ", eta)
        self.eta = eta
        self.gamma = gamma
        self.sampling_function = sampling_function
        self.reset(seed)
        
    def reset(self, seed, **kw):
        self.rng = np.random.default_rng(seed=seed)
        self.p = np.full(self.K, 1/self.K)
    
    def predict(self):
        exp_factor = np.full(self.K, 1/self.K)
        curr_p = (1 - self.gamma) * self.p + self.gamma * exp_factor
        return self.sampling_function(curr_p, self.M)
    
    def update(self, actions, reward):
        rew = max(reward)
        v_actions = np.zeros(self.K)
        v_actions[actions] = 1
        P_t = self.compute_inv(self.p)
        # P_t = np.zeros((self.K,self.K))
        estimate = P_t @ v_actions * rew
        factor = np.clip(self.eta * (estimate.T @ v_actions), None, 1e+02)
        self.p[actions] = self.p[actions] * np.exp(factor)
        fac = sum(self.p)
        self.p = self.p / fac
    
    def compute_inv(self, prob_dist):
        return np.linalg.pinv(np.outer(prob_dist, prob_dist))

    def compute_v_mat(self, prob_dist, iterations = 100):
        """
        Implement matrix geometric resempling
        """
        XX = np.zeros((len(prob_dist), len(prob_dist)))
        for _ in range(iterations):
            X = self.rng.binomial(1, p = prob_dist)
            XX += np.outer(X,X.T)
        return np.linalg.pinv(XX)
    
    def compute_mgr(self, prob_dist, iterations = 10, beta = 0.01):
        """
        Implement matrix geometric resempling
        """
        A_k = np.eye(len(prob_dist))
        cum_res_A = np.zeros((len(prob_dist), len(prob_dist)))
        for _ in range(iterations):
            if np.max(A_k) < 1e-5:
                break
            temp = self.rng.binomial(1, p = prob_dist)
            B_k = np.outer(temp,temp.T)
            np.matmul(A_k ,(np.eye(len(prob_dist)) - beta * B_k), out = A_k)
            cum_res_A += A_k
        return beta * np.eye(len(prob_dist)) - beta * cum_res_A
    
    def mgr_pt2(self, prob_dist, M = 10, beta = 0.01):
        size = len(prob_dist)
        A = np.zeros((M, size, size))
        temp = self.rng.binomial(1, p = prob_dist)
        estimate = np.outer(temp,temp)

        A[0] = np.identity(size) - beta * estimate

        for i in range(1, M):
            if np.max(A[i - 1]) < 1e-5:
                break
            temp = self.rng.binomial(1, p = prob_dist)
            estimate = np.outer(temp,temp)
            A[i] = A[i - 1] @ (np.identity(size) - beta * estimate)

        return beta * np.identity(size) + beta * np.sum(A, axis=0)

    def name(self):
        return f"ComBand"
    