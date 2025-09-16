import pandas as pd
import numpy as np

# Number of resources
n_samples = 20

# Generating random resource types and attributes
resource_types = ['instance', 'network', 'storage', 'database', 'load_balancer']
attributes = ['tags', 'instance_type', 'region', 'vpc', 'subnet']
resource_ids = np.random.choice([f'resource{i}' for i in range(1, 21)], size=n_samples)
types = np.random.choice(resource_types, size=n_samples)
attrs = np.random.choice(attributes, size=n_samples)

# Simulate modification frequency and modification times
change_frequency = np.random.randint(1, 10, size=n_samples)  # Frequency of changes (1-10)
modification_times = pd.to_datetime(np.random.choice(pd.date_range('2022-01-01', '2023-01-01', freq='H'), size=n_samples))

# Simulating conflict flag (1 for conflict, 0 for no conflict)
conflict_flags = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

# Create DataFrame
data = pd.DataFrame({
    'resource_id': resource_ids,
    'resource_type': types,
    'attribute': attrs,
    'change_frequency': change_frequency,
    'modification_time': modification_times,
    'conflict': conflict_flags
})

# Save the dataset to a CSV file
data.to_csv("simulated_IaC_data.csv", index=False)

# Show the first few rows of the dataset
print(data)
