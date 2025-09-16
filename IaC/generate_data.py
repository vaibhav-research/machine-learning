import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration for data generation - Aiming for better balance
NUM_SEQUENCES = 10000  # Increased number of sequences even further
MAX_SEQUENCE_LENGTH = 20
CONCURRENCY_WINDOW_SECONDS = 5
NUM_RESOURCES = 75
ATTRIBUTES_PER_RESOURCE = 7
USER_ROLES = ['developer', 'operator', 'admin']

# Define resource criticality and attribute sensitivity
resource_criticality = {f'resource_{i}': random.uniform(0.1, 0.9) for i in range(NUM_RESOURCES)}
attribute_sensitivity = {f'attr_{i}': random.uniform(0.1, 0.9) for i in range(ATTRIBUTES_PER_RESOURCE)}

def generate_sequence():
    resource_id = f'resource_{random.randint(0, NUM_RESOURCES - 1)}'
    sequence = []
    current_time = datetime.now() - timedelta(hours=random.randint(1, 150))
    related_changes = {}

    for _ in range(random.randint(5, MAX_SEQUENCE_LENGTH)):
        attribute = f'attr_{random.randint(0, ATTRIBUTES_PER_RESOURCE - 1)}'
        change_frequency = random.uniform(0.01, 0.8)
        change_magnitude = random.uniform(0.01, 1.0)
        user_role = random.choice(USER_ROLES)

        time_delta = timedelta(seconds=random.randint(1, 7200))
        current_time += time_delta

        is_concurrent = False
        related_resource_id = None
        for other_res, other_time in list(related_changes.items()):
            if current_time - other_time < timedelta(seconds=CONCURRENCY_WINDOW_SECONDS):
                is_concurrent = True
                related_resource_id = other_res
                break

        sequence.append({
            'resource_id': resource_id,
            'attribute': attribute,
            'change_frequency': change_frequency,
            'modification_time': current_time,
            'is_concurrent': is_concurrent,
            'related_resource_id': related_resource_id if is_concurrent else None,
            'criticality_score': resource_criticality[resource_id],
            'attribute_sensitivity_score': attribute_sensitivity[attribute],
            'change_magnitude': change_magnitude,
            'user_role': user_role
        })
        related_changes[resource_id] = current_time
        if random.random() < 0.15:
            resource_id = f'resource_{random.randint(0, NUM_RESOURCES - 1)}'
            related_changes = {k: v for k, v in related_changes.items() if v > current_time - timedelta(seconds=CONCURRENCY_WINDOW_SECONDS * 2)}

    return pd.DataFrame(sequence)

all_sequences = []
for _ in range(NUM_SEQUENCES):
    all_sequences.append(generate_sequence())

# Determine conflict after sequence - Stronger and more balanced conflict generation
data = []
for seq_df in all_sequences:
    if not seq_df.empty:
        last_event = seq_df.iloc[-1]
        conflict_probability = 0.30  # Significantly higher base probability

        # Even stronger multiplicative influence of risk factors
        if last_event['is_concurrent']:
            conflict_probability *= 4.0
        conflict_probability *= (1 + 2.5 * last_event['criticality_score'])
        conflict_probability *= (1 + 3.0 * last_event['attribute_sensitivity_score'])
        conflict_probability *= (1 + 3.5 * last_event['change_magnitude'])

        # Less dampening for non-critical changes
        if last_event['criticality_score'] < 0.3 and last_event['change_frequency'] < 0.2 and not last_event['is_concurrent']:
            conflict_probability *= 0.75

        # More aggressive generation of high-conflict scenarios
        if last_event['criticality_score'] > 0.6 and last_event['attribute_sensitivity_score'] > 0.6:
            conflict_probability = min(1.0, conflict_probability * 4.0)
        if last_event['change_magnitude'] > 0.7 and last_event['is_concurrent']:
            conflict_probability = min(1.0, conflict_probability * 3.5)

        # Introduce a random element to ensure some False labels
        if random.random() < 0.2:
            conflict = False
        else:
            conflict = random.random() < min(0.95, conflict_probability)

        for _, row in seq_df.iterrows():
            data.append({**row.to_dict(), 'conflict_after_sequence': conflict})

synthetic_data = pd.DataFrame(data)
synthetic_data.to_csv('synthetic_sequential_iac_data.csv', index=False)

print("Generated updated synthetic_sequential_iac_data.csv with a stronger focus on balance and conflict")