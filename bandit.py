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
from src.tree import TreeBandit
from tqdm.auto import tqdm

def replay_mabtree(df: pd.DataFrame, policy: pol.Policy, **kwargs):
    root = kwargs.get('root', None)

    history = pd.DataFrame()
    all_recs = []
    tree_bandit = TreeBandit(root, policy)

    policy.init(reset_model=True) # resets the policy
    for t, round in tqdm(df.groupby('createdAt'), leave=False):
        day_recs = []
        for u in tqdm(round['user'].unique(), leave=False):
            recs = [{'user': u, 'missionID': m.value['missionID']} for m in tree_bandit.select(n = (3, 1), user=u)]
            day_recs += recs

        all_recs += day_recs
        actions = round.merge(pd.DataFrame(day_recs), on=['user', 'missionID'], how='inner')
        history = pd.concat((history, actions), ignore_index=True)
        policy.update(train_df=history, day=t)
            
    return history, pd.DataFrame(all_recs)


def recommend(rank: list, missions: pd.DataFrame, n=1) -> list:
    ranked_missions = missions.loc[rank] # ranks missions
    top_missions = ranked_missions.groupby('kind', observed=True, sort=False).head(1) # selects the top mission of each kind
    return top_missions.index.tolist()[:n] # returns the top n missions


def replay(df: pd.DataFrame, policy, **kwargs):
    missions = kwargs.get('missions', None)

    history = pd.DataFrame()
    all_recs = []
    for t, round in tqdm(df.groupby('createdAt'), leave=False):
        day_recs = []
        for u in tqdm(round['user'].unique(), leave=False):
            rank = policy.select(u)
            recs = [{'user': u, 'missionID': rec} for rec in recommend(rank, missions, n=3)]
            day_recs += recs

        all_recs += day_recs
        actions = round.merge(pd.DataFrame(day_recs), on=['user', 'missionID'], how='inner')
        history = pd.concat((history, actions), ignore_index=True)
        policy.update(train_df=history, day=t)
            
    return history, pd.DataFrame(all_recs)

# %%
from collections import Counter
import numpy as np

def calculate_entropy(data):
    counter = Counter(data)
    n = len(data)
    return - sum([count/n * np.log2(count/n) for count in counter.values()])


def evaluate(policy, replay, **kwargs) -> tuple[pd.DataFrame, float]:
    rewards, recs = replay(df[['user', 'missionID', 'createdAt', 'reward']], policy, **kwargs)

    entropy = recs.groupby('user')['missionID'].agg(calculate_entropy).mean()
    rewards = rewards.groupby('createdAt')['reward'].sum().cumsum()

    return rewards, entropy


def repeated_evaluation(policies: dict, replay_fn, n=10, **kwargs):
    cumul_rewards = []
    entropies = []

    for _ in tqdm(range(n)):
        round_rewards = dict()
        round_entropies = dict()
        for name, policy in tqdm(policies.items(), leave=False):
            reward, entropy = evaluate(policy, replay_fn, **kwargs)
            round_rewards[name] = reward
            round_entropies[name] = entropy
    
        cumul_rewards.append(pd.concat(round_rewards))
        entropies.append(round_entropies)

    cumul_rewards = pd.concat(cumul_rewards, axis=1)
    entropies = pd.DataFrame(entropies)

    return cumul_rewards, entropies

# %%
import torch
import numpy
from src import models as mod
from src import baseline as base

torch.manual_seed(0)
numpy.random.seed(0)

policies = {
    'MABTree \u03B5-greedy MF': pol.MABTreeEpsilonGreedyML(model_class=mod.MF, num_users=n_users, num_missions=n_missions, embedding_dim=8),
    'MABTree \u03B5-greedy': pol.MABTreeEpsilonGreedy(),
    'MABTree UCB1': pol.MABTreeUCB(),
    'MABTree Random': pol.MABTReeRandom(),
}

cumul_rewards, entropies = repeated_evaluation(policies, replay_mabtree, root=root)
cumul_rewards.to_csv('./out/mabtree_rewards.csv', index=True)
entropies.to_csv('./out/mabtree_entropies.csv', index=True)

# %% [markdown]
# # Baseline

# %%
missions = df[['missionID', 'kind', 'TARGET']].drop_duplicates().set_index('missionID')
missions

# %%
import torch
import numpy
from src import baseline as base

torch.manual_seed(0)
numpy.random.seed(0)

policies = {
    'LinUCB': base.LinUCB(num_arms=n_missions, context_manager=base.ContextManager(n_users=n_users, n_features=n_missions)),
    'UCB1': base.UCB1(num_arms=n_missions),
    '\u03B5-greedy': base.EpsilonGreedy(num_arms=n_missions),
}

cumul_rewards, entropies = repeated_evaluation(policies, replay, missions=missions)
cumul_rewards.to_csv('./out/baseline_rewards.csv', index=True)
entropies.to_csv('./out/baseline_entropies.csv', index=True)


