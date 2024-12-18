# %%
import pandas as pd

df = pd.read_csv('./data/October_November_missions_full.csv')
df.rename(columns={'sub': 'user'}, inplace=True)
df['mission'] = df['kind'] + '_' + df['TARGET'].astype(str)

df = df[['user', 'mission', 'createdAt', 'kind', 'TARGET', 'performance']]
df = df.groupby('user').filter(lambda x: x['createdAt'].nunique() > 10)
df['createdAt'] = pd.to_datetime(df['createdAt']).dt.date

df['user'] = df['user'].astype('category').cat.codes
df['mission'] = df['mission'].astype('category')
df['missionID'] = df['mission'].cat.codes
df['kind'] = df['kind'].astype('category')

def reward(x):
    if x <= 1:
        return x
    return max(0, 2 - x**2)

df['reward'] = df['performance'].apply(reward)

df.sort_values(by=['createdAt', 'user'], inplace=True, ignore_index=True)
df

# %%
n_users = df['user'].max() + 1
n_missions = df['missionID'].max() + 1

n_users, n_missions

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

# %%
from src import policy as pol
from src import contextual as ctx
from src import models as mod
from src.tree import TreeBandit
from tqdm.auto import tqdm

def replay(df: pd.DataFrame, policy: pol.Policy, root: TreeNode):
    history = pd.DataFrame()
    tree_bandit = TreeBandit(root, policy)
    for t, round in tqdm(df.groupby('createdAt'), leave=False):
        day_recs = []
        for u in tqdm(round['user'].unique(), leave=False):
            policy.init()
            recs = [{'user': u, 'missionID': m.value['missionID']} for m in tree_bandit.select(n = (3, 1), user=u)]
            day_recs += recs

        actions = round.merge(pd.DataFrame(day_recs), on=['user', 'missionID'], how='inner')
        history = pd.concat((history, actions), ignore_index=True)
        policy.update(train_df=history, day=t)
            
    return history


def evaluate(policy) -> pd.DataFrame:
    rewards = replay(df[['user', 'missionID', 'createdAt', 'reward']], policy, root)
    rewards = rewards.groupby('createdAt')['reward'].sum().cumsum()

    return rewards

# %%
import torch
import numpy

torch.manual_seed(0)
numpy.random.seed(0)

policies = {
    '\u03B5-Greedy-AutoRec': pol.ModelEpsilonGreedy(model=mod.UserBasedAutoRec(n_users, n_missions, hidden_dim=16, dropout=0.1)),
    '\u03B5-Greedy-MF':      pol.ModelEpsilonGreedy(model=mod.MF(n_users, n_missions, embedding_dim=10)),
    '\u03B5-Greedy-MLP':     pol.ModelEpsilonGreedy(model=mod.MLP(n_users, n_missions, embedding_dim=16, hidden_dim=32, dropout=0.1)),
    '\u03B5-Greedy-Mean':    pol.MeanEpsilonGreedy(),
    'Random':                pol.RandomBandit(),
}

results = pd.concat([
    pd.concat({name: evaluate(policy) for name, policy in tqdm(policies.items(), leave=False)})
    for _ in tqdm(range(5))
], axis=1)

results
results.to_csv('./out/replay_results.csv', index=True)


