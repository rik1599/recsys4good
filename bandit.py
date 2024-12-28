# %%
import pandas as pd
import numpy as np

df = pd.read_csv('./data/bq-results-20241223-153559-1734968168534.csv')
df = df[~df['missionCatalog_id'].isin(['f1455712-fa2b-4feb-b7f9-ab8ddfa29e8d', 'd090d147-1ac9-4963-9c2b-1f5e663bad44'])]

df.rename(columns={'sub': 'user', 'missionCatalog_id': 'missionID'}, inplace=True)
df['mission'] = df['kind'] + '_' + df['TARGET'].astype(str)

df = df[['user', 'missionID', 'createdAt', 'kind', 'TARGET', 'performance']]
df = df.groupby('user').filter(lambda x: x['createdAt'].nunique() > 7)
df['createdAt'] = pd.to_datetime(df['createdAt']).dt.date

df['user'] = df['user'].astype('category').cat.codes
df['missionID'] = df['missionID'].astype('category').cat.codes
df['kind'] = df['kind'].astype('category')

def reward(x):
    if x <= 1:
        return x
    return max(0, 2 - np.pow(x, 2))

df['reward'] = df['performance'].apply(reward)

df.sort_values(by=['createdAt', 'user'], inplace=True, ignore_index=True)
df

# %%
n_users = df['user'].max() + 1
n_missions = df['missionID'].max() + 1

print(n_users, n_missions)

# %%
from src.tree import TreeNode

missions = df[['missionID', 'kind', 'TARGET']].drop_duplicates()

root = TreeNode('root')
for name, round in missions.groupby('kind', observed=True):
    node = TreeNode(name)
    root.add_child(node)
    for _, mission in round.iterrows():
        node.add_child(TreeNode(mission.to_dict()))

print(root)

missions = missions.set_index('missionID')
print(missions)

# %%
from src import policy as pol
from src.tree import TreeBandit
from tqdm.auto import tqdm
 
class MABRecommender:
    def __init__(self, policy: pol.Policy):
        self.policy = policy

    def recommend(self):
        pass

class MABTreeRecommender(MABRecommender):
    def __init__(self, policy: pol.Policy, root: TreeNode):
        super().__init__(policy)
        self.root = root
        self.tree_bandit = TreeBandit(self.root, self.policy)
    
    def recommend(self, user, n=(3, 1)):
        recs = self.tree_bandit.select(n=n, user=user)
        return [m.value['missionID'] for m in recs]

class MABPolicyRecommender(MABRecommender):
    def __init__(self, policy: pol.Policy, missions: pd.DataFrame):
        super().__init__(policy)
        self.missions = missions
    
    def recommend(self, user, n=3):
        rank = self.policy.select(user)
        ranked_missions = self.missions.loc[rank] # ranks missions
        top_missions = ranked_missions.groupby('kind', observed=True, sort=False).head(1) # selects the top mission of each kind
        return top_missions.index.tolist()[:n] # returns the top n missions


def replay(df: pd.DataFrame, recommeder: MABRecommender):
    history = pd.DataFrame()
    policy_recommendations = []

    recommeder.policy.init()
    for t, round in tqdm(df.groupby('createdAt'), leave=False):
        day_recs = []
        for u in tqdm(round['user'].unique(), leave=False):
            user_recs = [{'date': t, 'user': u, 'missionID': m} for m in recommeder.recommend(u)]
            day_recs += user_recs
        
        policy_recommendations += day_recs
        actions = round.merge(pd.DataFrame(day_recs), on=['user', 'missionID'], how='inner')
        history = pd.concat((history, actions), ignore_index=True)
        recommeder.policy.update(train_df=history, day=t)
    
    return history, pd.DataFrame(policy_recommendations)

# %%
def per_day_entropy(df: pd.DataFrame):
    def entropy(x):
        _, counts = np.unique(x, return_counts=True)
        return - np.sum(counts / len(x) * np.log2(counts / len(x)))

    return df.groupby('date')['missionID'].apply(entropy).mean()

def avg_coverage(df: pd.DataFrame):
    return df.groupby('user')['missionID'].nunique().apply(lambda x: x / n_missions).mean()

def cumulative_reward(df: pd.DataFrame):
    return df.groupby('createdAt')['reward'].sum().cumsum()

def evaluate(recommender: MABRecommender):
    history, policy_recommendations = replay(df[['user', 'missionID', 'createdAt', 'reward']], recommender)

    return cumulative_reward(history), avg_coverage(policy_recommendations), per_day_entropy(policy_recommendations)

# %%
import torch
import numpy
from src import models as mod
from src import baseline as base

torch.manual_seed(0)
numpy.random.seed(0)

policies = {
    'MABTree Random': MABTreeRecommender(pol.MABTReeRandom(), root),
    'MABTree UCB': MABTreeRecommender(pol.MABTreeUCB(), root),
    'MABTree 0.1-greedy': MABTreeRecommender(pol.MABTreeEpsilonGreedy(0.1), root),
    'MABTree 0.1-greedy MF': MABTreeRecommender(pol.MABTreeEpsilonGreedyML(model_class=mod.MF, num_users=n_users, num_missions=n_missions, embedding_dim=8), root),
    '0.1-greedy': MABPolicyRecommender(base.EpsilonGreedy(num_arms=n_missions, epsilon=0.1), missions),
    'UCB': MABPolicyRecommender(base.UCB1(num_arms=n_missions), missions),
    'LinUCB': MABPolicyRecommender(base.LinUCB(num_arms=n_missions, context_manager=base.ContextManager(n_users, n_missions)), missions),
}

# %%
metrics = [
    {name: evaluate(recommender) for name, recommender in tqdm(policies.items(), leave=False)}
    for _ in tqdm(range(10))
]

# %%
# Cumulated reward dataframe

pd.concat([
    pd.concat({
        name: reward for name, (reward, _, _) in metric.items()
    }) for metric in metrics
], axis=1).to_csv('./out/cumulated_reward.csv', index=True)

# %%
# Coverage dataframe

pd.DataFrame([
    {name: coverage for name, (_, coverage, _) in metric.items()}
    for metric in metrics
]).to_csv('./out/coverage.csv', index=True)

# %%
# Inter-list diversity dataframe

pd.DataFrame([
    {name: diversity for name, (_, _, diversity) in metric.items()}
    for metric in metrics
]).to_csv('./out/diversity.csv', index=True)


