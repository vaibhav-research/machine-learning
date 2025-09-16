def compute_cost(current_instances, cost_params):
    """Calculate the cost based on current instances and scaling decisions."""
    scaling_cost = cost_params['C_up'] * (current_instances - cost_params['prev_c'])
    base_cost = cost_params['C'] * current_instances
    return base_cost + scaling_cost
