import pytest
from app.scaling_logic import autoscale

# Test comparison with threshold-based scaling
def test_threshold_based_scaling():
    thresholds = {'T_rho': 0.8, 'T_P': 0.3, 'T_W': 0.5}
    cost_params = {'C': 5, 'C_up': 10, 'prev_c': 10}
    sla_params = {'R': 100, 'P_s': 50}
    alpha = 0.7
    beta = 0.3

    # Simple threshold-based logic
    lambda_rate = 120
    mu = 10
    c = 10

    # Compare with the predictive scaling decision
    decision_game_theory = autoscale(lambda_rate, mu, c, thresholds, cost_params, sla_params, alpha, beta)

    # Add logic for a threshold-based decision (simplified)
    decision_threshold_based = "Scale Up" if lambda_rate > c * mu else "Maintain"
    
    assert decision_game_theory != decision_threshold_based
