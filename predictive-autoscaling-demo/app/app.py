from datetime import datetime, timedelta
import random
from flask import Flask, jsonify, request
from scaling_logic import autoscale, compute_cost, extract_features

app = Flask(__name__)

# Thresholds for Nash equilibrium and scaling decisions
thresholds = {
    "equilibrium_threshold": 0.05,  # Threshold to decide when to scale
    "cost_threshold": 10  # Threshold for cost change
}

def generate_traffic_data():
    traffic_data = []
    current_time = datetime.now()

    # Generate data for the past 30 days
    for i in range(30):
        time_point = current_time - timedelta(days=i)
        requests = random.randint(50, 500)  # Simulated number of requests
        cpu_utilization = random.uniform(10.0, 90.0)  # Simulated CPU utilization in percentage
        traffic_data.append({
            'time': time_point.strftime('%Y-%m-%d %H:%M:%S'),
            'requests': requests,
            'cpu_utilization': cpu_utilization
        })

    return traffic_data

def get_traffic_load(traffic_data):
    # Convert traffic data to numerical values (e.g., time between events)
    if len(traffic_data) < 2:
        return 0  # Not enough data to calculate traffic load
    traffic_timestamps = [datetime.strptime(t['time'], "%Y-%m-%d %H:%M:%S") for t in traffic_data]
    time_diff = (traffic_timestamps[-1] - traffic_timestamps[0]).total_seconds()
    return time_diff if time_diff > 0 else 0


@app.route('/scale', methods=['GET'])
@app.route('/scale', methods=['GET'])
def scale():
    # Fetch current instances from the AWS Auto Scaling Group
    current_instances = 1  # This is a placeholder; you may fetch this dynamically from AWS ASG API

    # Get traffic data from /traffic endpoint (should be updated dynamically)
    traffic_data = generate_traffic_data()  # For demo purposes, you can call this function directly
    
    print(f"Fetched Traffic Data: {traffic_data}")  # Log traffic data to debug

    # If no traffic data is available, scale down or take no action
    if len(traffic_data) == 0:
        scaling_decision = "No Action"
        new_instance_count = current_instances
    else:
        # Process traffic data and make scaling decision
        current_traffic = get_traffic_load(traffic_data)  # Convert traffic data to numerical value
        
        # Define thresholds for scaling decision
        thresholds = {
            "equilibrium_threshold": 1.0,  # Example threshold for Nash equilibrium
            "cost_threshold": 5.0         # Example threshold for cost-based decision
        }

        # Make scaling decision based on current traffic and current instances
        scaling_decision, new_instance_count = autoscale(current_traffic, current_instances, thresholds)

    return jsonify({
        "scaling_decision": scaling_decision,
        "new_instance_count": new_instance_count
    })

@app.route('/cost', methods=['GET'])
def get_cost():
    features = extract_features()
    current_instances = features["current_instances"]
    
    # Calculate the cost based on current instance count
    cost_current = compute_cost(current_instances)
    
    response = {
        "current_instances": current_instances,
        "current_cost": cost_current
    }
    
    return jsonify(response), 200

@app.route('/traffic', methods=['GET'])
def get_traffic():
    # Generate traffic data for the past 30 days
    traffic_data = generate_traffic_data()
    return jsonify({
        'traffic_data': traffic_data
    })


if __name__ == "__main__":
    app.run(debug=True)
