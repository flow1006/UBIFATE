import torch
import torch.nn as nn
import torch.nn.functional as F


class IntelligentCarInsuranceModel(nn.Module):
    def __init__(self):
        super(IntelligentCarInsuranceModel, self).__init__()
        # Define the layers and their configurations for the Intelligent Car Insurance model
        self.age_layer = nn.Linear(1, 10)
        self.driving_experience_layer = nn.Linear(1, 10)
        self.vehicle_type_layer = nn.Linear(3, 10)  # Assuming one-hot encoding for vehicle type
        self.speeding_incidents_layer = nn.Linear(1, 10)
        self.coverage_type_layer = nn.Linear(3, 10)  # Assuming one-hot encoding for coverage type
        self.claims_history_layer = nn.Linear(1, 10)
        self.final_layer = nn.Linear(60, 1)  # Combining all features for final prediction

    def forward(self, age, driving_experience, vehicle_type, speeding_incidents, coverage_type, claims_history):
        age_embedding = F.relu(self.age_layer(age))
        driving_experience_embedding = F.relu(self.driving_experience_layer(driving_experience))
        vehicle_type_embedding = F.relu(self.vehicle_type_layer(vehicle_type))
        speeding_incidents_embedding = F.relu(self.speeding_incidents_layer(speeding_incidents))
        coverage_type_embedding = F.relu(self.coverage_type_layer(coverage_type))
        claims_history_embedding = F.relu(self.claims_history_layer(claims_history))

        # Concatenate all embeddings
        x = torch.cat([
            age_embedding, driving_experience_embedding, vehicle_type_embedding,
            speeding_incidents_embedding, coverage_type_embedding, claims_history_embedding
        ], dim=1)

        # Apply final layer for prediction
        x = self.final_layer(x)

        return x
