import numpy as np

class LinUCB:
    def __init__(self, n_arms, n_features, alpha=1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
    
    def select_arm(self, context: np.ndarray) -> int:
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta: np.ndarray = A_inv.dot(self.b[arm])
            p[arm] = theta.dot(context) + self.alpha * np.sqrt(context.dot(A_inv).dot(context))
        return np.argmax(p)
    
    def update(self, chosen_arm, reward, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context

# Example usage
if __name__ == "__main__":
    n_arms = 5
    n_features = 10
    alpha = 1.0
    linucb = LinUCB(n_arms, n_features, alpha)
    
    for _ in range(200):
    # Simulate some data
        context = np.random.rand(n_features)
        chosen_arm = linucb.select_arm(context)
        reward = np.random.rand()  # Simulated reward
        linucb.update(chosen_arm, reward, context)
    
        print(f"Chosen arm: {chosen_arm}, Reward: {reward:.4f}")