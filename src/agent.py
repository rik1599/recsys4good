import random
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MissionRecommender:
    def __init__(self, missions, model, epsilon):
        """
        Initialize the recommender.

        Args:
        missions (list of dict): List of mission objects, each with 'ID', 'type', and 'target'.
        model (torch.nn.Module): PyTorch module to rank missions based on user history.
        epsilon (float): Probability of selecting a random mission.
        """
        self.missions = missions
        self.model = model
        self.epsilon = epsilon

    def recommend(self, user_history, num_recommendations):
        """
        Generate a set of mission recommendations based on the policy.

        Args:
        user_history (list of tuples): User's past interactions [(mission_id, outcome), ...].
        num_recommendations (int): Number of missions to recommend.

        Returns:
        list of dict: Recommended missions.
        """
        recommendations = []
        used_mission_ids = self._get_used_missions(user_history)
        assigned_types = set()

        while len(recommendations) < num_recommendations:
            if self._use_random_selection():
                self._add_random_mission(recommendations, assigned_types)
            else:
                self._add_ranked_mission(recommendations, user_history, used_mission_ids, assigned_types)

        return recommendations

    def _get_used_missions(self, user_history):
        """
        Extract the IDs of missions with positive outcomes from the user's history.

        Args:
        user_history (list of tuples): User's past interactions.

        Returns:
        set: Mission IDs with positive outcomes.
        """
        return {m[0] for m in user_history if m[1] > 0}

    def _use_random_selection(self):
        """
        Decide whether to select a mission randomly based on epsilon.

        Returns:
        bool: True if random selection is chosen, False otherwise.
        """
        return random.random() < self.epsilon

    def _add_random_mission(self, recommendations, assigned_types):
        """
        Add a random mission to the recommendations if possible.

        Args:
        recommendations (list): Current recommendation list.
        assigned_types (set): Set of already assigned types.
        """
        random_mission = self._select_random_mission(assigned_types)
        if random_mission:
            recommendations.append(random_mission)
            assigned_types.add(random_mission["type"])

    def _add_ranked_mission(self, recommendations, user_history, used_mission_ids, assigned_types):
        """
        Add the highest-ranked mission to the recommendations.

        Args:
        recommendations (list): Current recommendation list.
        user_history (list of tuples): User's past interactions.
        used_mission_ids (set): IDs of missions to exclude from recommendations.
        assigned_types (set): Set of already assigned types.
        """
        ranked_missions = self._rank_missions(user_history, used_mission_ids)
        for mission in ranked_missions:
            if mission["type"] not in assigned_types:
                recommendations.append(mission)
                assigned_types.add(mission["type"])
                self._replace_existing_random_of_same_type(recommendations, mission)
                break

    def _replace_existing_random_of_same_type(self, recommendations, mission):
        """
        Replace any random mission of the same type in recommendations with the ranked mission.

        Args:
        recommendations (list): Current recommendation list.
        mission (dict): The ranked mission to be added.
        """
        recommendations[:] = [
            rec for rec in recommendations if rec["type"] != mission["type"] or rec == mission
        ]

    def _select_random_mission(self, assigned_types):
        """
        Select a random mission that is not of an already assigned type.

        Args:
        assigned_types (set): Types already assigned.

        Returns:
        dict or None: Random mission or None if no valid mission found.
        """
        available_missions = [m for m in self.missions if m["type"] not in assigned_types]
        return random.choice(available_missions) if available_missions else None

    def _rank_missions(self, user_history, used_mission_ids):
        """
        Query the model for a ranked list of missions, excluding used ones.

        Args:
        user_history (list of tuples): User's past interactions.
        used_mission_ids (set): Set of mission IDs to exclude.

        Returns:
        list of dict: Ranked list of missions.
        """
        # Convert user history to PyTorch tensor
        history_tensor = torch.from_numpy(user_history).view(1, -1, 2).to(DEVICE)
        
        # Get mission scores from the model
        with torch.no_grad():
            mission_scores = self.model(history_tensor).squeeze(0)  # Assume model outputs a (num_missions,) tensor

        # Rank missions by score
        mission_ranking = torch.argsort(mission_scores, descending=True).tolist()
        
        # Exclude missions with positive outcomes
        ranked_missions = [
            m for idx in mission_ranking
            for m in self.missions
            if m["ID"] == idx and m["ID"] not in used_mission_ids
        ]
        return ranked_missions
