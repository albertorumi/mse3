import numpy as np
import pandas as pd

class Movielens:
    def __init__(self, M, path_data = '/data/fast/rumidata/ml-25m/ratings.csv', seed = 42, test = False):
        """
        Reward based initialization
        """
        self.rng = np.random.default_rng(seed=seed)
        self.M = M

        df = pd.read_csv(path_data)
        if test:
            df = df.head(10_000)
        self.users = df.userId.unique()
        movies = df.movieId.unique()
        # Pre-process: binarize ratings
        threshold = 3.5
        df['binarized_rate'] = df['rating'].apply(lambda x: 1 if x >= threshold else 0)
        self.pref_dict = {
            user : set() for user in self.users
        }
        prefdf = df[df['binarized_rate'] == 1]
        for r in prefdf.values:
            u = int(r[0])
            m = int(r[1])
            self.pref_dict[u].add(m)
        
        self.K = len(movies)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed=seed)

    def get_reward(self, action):
        assert len(action) <= self.M, f"Error in action size: {len(action)}, actions available: {self.M}"
        curr_user = self.rng.choice(self.users)
        reward = [int(el in self.pref_dict[curr_user]) for el in action]
        return reward
    
    def name(self):
        return "Movielens"