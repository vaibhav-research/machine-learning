import boto3
import numpy as np
from datetime import datetime, timedelta

# Initialize AWS client for Auto Scaling
asg_client = boto3.client('autoscaling', region_name='us-east-2')

# Constants for the demo
COST_PER_INSTANCE = 10  # Cost per instance (arbitrary value)
PREDICTION_WINDOW = 5  # Number of past data points to consider for prediction
TRAFFIC_QUEUE_CAPACITY = 1000  # Maximum queue capacity (for queuing theory)

# Define your Auto Scaling Group name
ASG_NAME = 'use2-preprod-test-workers-resque-low-priority-idempotent-exchanger'  # Replace with your ASG name

# Fetch ASG data from AWS
def fetch_asg_data():
    # Get desired capacity and actual instances in the Auto Scaling Group (ASG)
    asg_data = asg_client.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])
    asg_group = asg_data['AutoScalingGroups'][0]

    # Extract relevant data: desired capacity and actual instance counts
    desired_capacity = asg_group['DesiredCapacity']
    current_instance_count = len(asg_group['Instances'])  # Correct way to get number of instances
    
    # Return number of instances and scaling events (simulated for this demo)
    scaling_events = fetch_scaling_events()

    return desired_capacity, current_instance_count, scaling_events

# Fetch historical scaling events from the Auto Scaling Group
def fetch_scaling_events():
    try:
        scaling_history = asg_client.describe_scaling_activities(
            AutoScalingGroupName=ASG_NAME,
            MaxRecords=50  # Fetch the latest 50 scaling events
        )

        if 'ScalingActivities' not in scaling_history:
            print("No scaling activities found.")
            return []  # Return empty list if no scaling activities are found

        events = []
        for activity in scaling_history['ScalingActivities']:
            event_time = activity['StartTime']
            event_type = activity['StatusCode']
            events.append({
                'time': event_time,
                'event': event_type
            })
        
        return events

    except Exception as e:
        print(f"Error fetching scaling events: {e}")
        return []  # Return empty list on error

# Moving average calculation for predicted load
def moving_average(data, window_size=5):
    if len(data) == 0:
        return 0  # If no data, return 0 as a safe default
    if len(data) < window_size:
        return np.mean(data)  # Return mean of all data if less than window size
    return np.mean(data[-window_size:])  # Otherwise, return mean of the last `window_size` data points

# Function to simulate scaling decision with Nash Equilibrium
def autoscale(current_traffic, current_instances, thresholds):
    # Fetch ASG data
    desired_capacity, current_instance_count, scaling_events = fetch_asg_data()
    
    traffic_history = [event['time'].timestamp() for event in scaling_events]  # Convert timestamps to UNIX timestamps
    
    if len(traffic_history) == 0:
        predicted_load = 0  # Default if no traffic data is available
    else:
        predicted_load = moving_average(traffic_history)
    
    print(f"Predicted Load: {predicted_load}")

    # Compute utility and cost for scaling decisions (Game theory)
    utility_current = current_instances * 1  # Assume each instance has a utility value of 1
    utility_up = (current_instances + 1) * 1
    utility_down = (current_instances - 1) * 1 if current_instances > 1 else 0

    cost_current = current_instances * COST_PER_INSTANCE
    cost_up = (current_instances + 1) * COST_PER_INSTANCE
    cost_down = (current_instances - 1) * COST_PER_INSTANCE if current_instances > 1 else 0

    # Compute utility and cost difference
    utility_diff_up = utility_up - utility_current
    cost_diff_up = cost_up - cost_current

    utility_diff_down = utility_down - utility_current
    cost_diff_down = cost_down - cost_current

    # Nash Equilibrium concept: The optimal scaling point is when utility and cost are balanced
    # Here we consider both scaling up and scaling down options
    equilibrium_utility_cost_diff_up = abs(utility_diff_up - cost_diff_up)
    equilibrium_utility_cost_diff_down = abs(utility_diff_down - cost_diff_down)

    print(f"Utility Diff (Up): {utility_diff_up}, Cost Diff (Up): {cost_diff_up}")
    print(f"Utility Diff (Down): {utility_diff_down}, Cost Diff (Down): {cost_diff_down}")
    print(f"Equilibrium Utility-Cost Diff (Up): {equilibrium_utility_cost_diff_up}")
    print(f"Equilibrium Utility-Cost Diff (Down): {equilibrium_utility_cost_diff_down}")

    # Use desired_capacity and current_instance_count to make better scaling decisions
    if desired_capacity > current_instance_count:
        scaling_decision = "Scale Up"  # Scale up if desired capacity is greater than current instances
        current_instance_count += 1
    elif desired_capacity < current_instance_count:
        scaling_decision = "Scale Down"  # Scale down if desired capacity is less than current instances
        current_instance_count -= 1
    else:
        # No scaling needed if desired capacity matches current instances
        if equilibrium_utility_cost_diff_up < thresholds["equilibrium_threshold"] and equilibrium_utility_cost_diff_down < thresholds["equilibrium_threshold"]:
            scaling_decision = "No Action"
        elif predicted_load > current_traffic and utility_diff_up > 0 and cost_diff_up < thresholds["cost_threshold"]:
            scaling_decision = "Scale Up"
            current_instance_count += 1  # Increment instance count
        elif predicted_load < current_traffic and utility_diff_down < 0 and cost_diff_down > thresholds["cost_threshold"]:
            scaling_decision = "Scale Down"
            current_instance_count -= 1  # Decrement instance count
        else:
            scaling_decision = "No Action"

    # Apply queuing theory to estimate the potential load on each instance
    load_per_instance = current_traffic / current_instance_count if current_instance_count > 0 else 0
    print(f"Load per instance: {load_per_instance}")

    # If the load exceeds a threshold or is too high, scale up or down based on queuing theory
    if load_per_instance > TRAFFIC_QUEUE_CAPACITY and scaling_decision == "No Action":
        scaling_decision = "Scale Up"
        current_instance_count += 1  # Add an instance to balance the load
    elif load_per_instance < TRAFFIC_QUEUE_CAPACITY / 2 and scaling_decision == "No Action":
        scaling_decision = "Scale Down"
        current_instance_count -= 1  # Remove an instance to avoid over-provisioning

    # Make sure `current_instance_count` never goes below 1
    current_instance_count = max(current_instance_count, 1)

    print(f"Scaling Decision: {scaling_decision}, New Instance Count: {current_instance_count}")
    
    return {
        "new_instance_count": current_instance_count,
        "scaling_decision": scaling_decision
    }

# Cost calculation function
def compute_cost(instances):
    return instances * COST_PER_INSTANCE

# Feature engineering to extract necessary fields for the app
def extract_features():
    # Fetch ASG data
    desired_capacity, current_instances, scaling_events = fetch_asg_data()

    # Feature engineering: Extract useful data
    traffic_data = [event['time'].strftime("%Y-%m-%d %H:%M:%S") for event in scaling_events]
    scaling_decisions = [event['event'] for event in scaling_events]

    # Return the engineered features
    return {
        "desired_capacity": desired_capacity,
        "current_instances": current_instances,
        "scaling_decisions": scaling_decisions,
        "traffic_data": traffic_data
    }

# Example usage (testing the functionality)
if __name__ == "__main__":
    thresholds = {
        "equilibrium_threshold": 5,
        "cost_threshold": 30
    }

    # Simulate traffic data for testing
    current_traffic = 800
    current_instances = 2
    
    result = autoscale(current_traffic, current_instances, thresholds)
    print(result)
