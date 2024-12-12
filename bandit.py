# %%
import pandas as pd

df = pd.concat((
    pd.read_csv('./data/October_missions_full.csv'),
    pd.read_csv('./data/November_1stW_missions_full.csv')
), ignore_index=True)

df['mission'] = df['type'] + '_' + df['target'].astype(str)

df = df[['user', 'mission', 'createdAtT', 'type', 'target', 'performance']]
df['createdAtT'] = pd.to_datetime(df['createdAtT'], unit='ms').dt.date
df = df.groupby('user').filter(lambda x: len(x['createdAtT'].unique()) > 1)

df['user'] = df['user'].astype('category').cat.codes
df['mission'] = df['mission'].astype('category')
df['missionID'] = df['mission'].cat.codes
df['type'] = df['type'].astype('category')

def reward(x):
    if x <= 1:
        return x
    return max(0, 2 - x**2)

df['reward'] = df['performance'].apply(reward)
df.rename(columns={'createdAtT': 'date'}, inplace=True)

df.sort_values(by=['date', 'user'], inplace=True, ignore_index=True)
df

# %%
n_users = df['user'].max() + 1
n_missions = df['missionID'].max() + 1

n_users, n_missions

# %%
from src.tree import TreeNode

missions = df[['missionID', 'type', 'target']].drop_duplicates()
root = TreeNode('root')
for name, round in missions.groupby('type', observed=True):
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
    for t, round in tqdm(df.groupby('date'), leave=False):
        day_recs = []
        for u in tqdm(round['user'].unique(), leave=False):
            policy.init()
            recs = [{'user': u, 'missionID': m.value['missionID']} for m in tree_bandit.select(n = (3, 1), user=u)]
            day_recs += recs

        actions = round.merge(pd.DataFrame(day_recs), on=['user', 'missionID'], how='inner')
        history = pd.concat((history, actions), ignore_index=True)
        policy.update(train_df=history, day=t)
            
    return history

# %%
def evaluate(policy) -> pd.DataFrame:
    rewards = replay(df[['user', 'missionID', 'date', 'reward']], policy, root)
    rewards = rewards.groupby('date')['reward'].sum().cumsum()

    return rewards

# %%
import torch
import numpy

torch.manual_seed(0)
numpy.random.seed(0)

policies = {
    'E-Greedy-AutoRec': pol.ModelEpsilonGreedy(model=mod.UserBasedAutoRec(n_users, n_missions, hidden_dim=32, dropout=0.1)),
    'LinUCB': ctx.LinUCB(n_users, n_missions, n_missions, ctx.ContextManager(n_users=n_users, features=df['mission'].cat.categories)),
    'Random': pol.RandomBandit(),
    'E-Greedy-Mean': pol.MeanEpsilonGreedy(),
}

results = pd.concat([
    pd.concat({name: evaluate(policy) for name, policy in tqdm(policies.items(), leave=False)})
    for _ in tqdm(range(1))
], axis=1)

results
results.to_csv('./out/replay_results.csv', index=True)