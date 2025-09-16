import pytest
from app.scaling_logic import autoscale, compute_utility

# Test utility function
def test_compute_utility():
    sla_params = {'R': 100, 'P_s': 50}
    cost_params = {'C': 5, 'C_up': 10, 'prev_c': 10}
    alpha = 0.7
    beta = 0.3
    utility = compute_utility(10, 120, 10, sla_params, cost_params, alpha, beta)
    assert utility > 0

# Test autoscaling decision logic
def test_autoscale():
    sla_params = {'R': 100, 'P_s': 50}
    cost_params = {'C': 5, 'C_up': 10, 'prev_c': 10}
    thresholds = {'T_rho': 0.8, 'T_P': 0.3, 'T_W': 0.5}
    alpha = 0.7
    beta = 0.3

    # Test with random values
    scaling_decision = autoscale(120, 10, 10, thresholds, cost_params, sla_params, alpha, beta)
    assert scaling_decision in ["Scale Up", "Scale Down", "Maintain"]
