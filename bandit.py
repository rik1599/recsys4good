# %%
import pandas as pd

df = pd.read_csv('./data/October_November_missions_full.csv')
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
    return max(0, 2 - x**2)

df['reward'] = df['performance'].apply(reward)

df.sort_values(by=['createdAt', 'user'], inplace=True, ignore_index=True)
df

# %%
n_users = df['user'].max() + 1
n_missions = df['missionID'].max() + 1

print(n_users, n_missions)

# %% [markdown]
# # MABTree

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
    '\u03B5-Greedy-MF':      pol.ModelEpsilonGreedy(model=mod.MF(n_users, n_missions, embedding_dim=8)),
    '\u03B5-Greedy-NeuMF':   pol.ModelEpsilonGreedy(model=mod.NeuMF(n_users, n_missions, embedding_dim=8, hidden_dim=8, dropout=0.1)),
    '\u03B5-Greedy-Mean':    pol.MeanEpsilonGreedy(),
    'Random':                pol.RandomBandit(),
}

results = pd.concat([
    pd.concat({name: evaluate(policy) for name, policy in tqdm(policies.items(), leave=False)})
    for _ in tqdm(range(5))
], axis=1)

results
results.to_csv('./out/replay_results.csv', index=True)

# %% [markdown]
# # Baseline

# %%
missions = df[['missionID', 'kind', 'TARGET']].drop_duplicates().set_index('missionID')
missions

# %%
from src import contextual as ctx
from tqdm.auto import tqdm

def recommend(rank: list, missions: pd.DataFrame, n=1) -> list:
    ranked_missions = missions.loc[rank] # ranks missions
    top_missions = ranked_missions.groupby('kind', observed=True, sort=False).head(1) # selects the top mission of each kind
    return top_missions.index.tolist()[:n] # returns the top n missions

def replay(df: pd.DataFrame, policy: ctx.LinUCB):
    history = pd.DataFrame()
    for t, round in tqdm(df.groupby('createdAt'), leave=False):
        day_recs = []
        for u in tqdm(round['user'].unique(), leave=False):
            rank = policy.select(u)
            recs = [{'user': u, 'missionID': rec} for rec in recommend(rank, missions, n=3)]
            day_recs += recs

        actions = round.merge(pd.DataFrame(day_recs), on=['user', 'missionID'], how='inner')
        history = pd.concat((history, actions), ignore_index=True)
        policy.update(train_df=history, day=t)
            
    return history


def evaluate(policy) -> pd.DataFrame:
    rewards = replay(df[['user', 'missionID', 'createdAt', 'reward']], policy)
    rewards = rewards.groupby('createdAt')['reward'].sum().cumsum()

    return rewards

# %%
import torch
import numpy

torch.manual_seed(0)
numpy.random.seed(0)

policies = {
    'LinUCB': ctx.LinUCB(n_missions, context_manager=ctx.ContextManager(n_users, n_missions)),
}

results = pd.concat([
    pd.concat({name: evaluate(policy) for name, policy in tqdm(policies.items(), leave=False)})
    for _ in tqdm(range(5))
], axis=1)

results
results.to_csv('./out/replay_results_baseline.csv', index=True)


