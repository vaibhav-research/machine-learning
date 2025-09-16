import random

class TrafficSimulator:
    def __init__(self, initial_lambda_rate=100):
        self.lambda_rate = initial_lambda_rate
    
    def generate_traffic(self):
        """Simulate incoming traffic dynamically."""
        # Lambda rate varies randomly to simulate varying load
        self.lambda_rate += random.randint(100, 500)  # Random increase in traffic
        return self.lambda_rate
