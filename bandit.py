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

missions = missions.set_index('missionID')

# %%
from src import policy as pol
from src.tree import TreeBandit
from tqdm.auto import tqdm

def recommend(rank: list, missions: pd.DataFrame, n=1) -> list:
    ranked_missions = missions.loc[rank] # ranks missions
    top_missions = ranked_missions.groupby('kind', observed=True, sort=False).head(1) # selects the top mission of each kind
    return top_missions.index.tolist()[:n] # returns the top n missions


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
            
    return history, all_recs


def replay(df: pd.DataFrame, policy: pol.Policy, **kwargs):
    missions = kwargs.get('missions', None)

    history = pd.DataFrame()
    all_recs = []
    policy.init(reset_model=True) # resets the policy
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
            
    return history, all_recs

# %%
from collections import Counter
import numpy as np

def calculate_entropy(data):
    counter = Counter(data)
    n = len(data)
    return - sum([count/n * np.log2(count/n) for count in counter.values()])


def calculate_coverage(data):
    return len(set(data)) / n_missions


def evaluate(policy, replay, **kwargs) -> tuple[pd.DataFrame, float]:
    rewards, recs = replay(df[['user', 'missionID', 'createdAt', 'reward']], policy, **kwargs)

    recs = pd.DataFrame(recs)
    entropy = recs.groupby('user')['missionID'].agg(calculate_entropy).mean()
    coverage = recs.groupby('user')['missionID'].agg(calculate_coverage).mean()
    rewards = rewards.groupby('createdAt')['reward'].sum().cumsum()

    return rewards, entropy, coverage


def repeated_evaluation(policies: dict, replay_fn, n=10, **kwargs) -> list[pd.DataFrame]:
    metrics = [[], [], []]

    for _ in tqdm(range(n)):
        round_rewards = dict()
        round_entropies = dict()
        round_coverages = dict()
        for name, policy in tqdm(policies.items(), leave=False):
            round_rewards[name], round_entropies[name], round_coverages[name] = evaluate(policy, replay_fn, **kwargs)
    
        metrics[0].append(pd.concat(round_rewards))
        metrics[1].append(round_entropies)
        metrics[2].append(round_coverages)

    metrics[0] = pd.concat(metrics[0], axis=1)
    metrics[1] = pd.DataFrame(metrics[1])
    metrics[2] = pd.DataFrame(metrics[2])

    return metrics

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

metrics = repeated_evaluation(policies, replay_mabtree, root=root)
metrics[0].to_csv('./out/mabtree_rewards.csv', index=True)
metrics[1].to_csv('./out/mabtree_entropies.csv', index=True)
metrics[2].to_csv('./out/mabtree_coverages.csv', index=True)

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

metrics = repeated_evaluation(policies, replay, missions=missions)
metrics[0].to_csv('./out/baseline_rewards.csv', index=True)
metrics[1].to_csv('./out/baseline_entropies.csv', index=True)
metrics[2].to_csv('./out/baseline_coverages.csv', index=True)


