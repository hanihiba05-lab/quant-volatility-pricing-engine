import unittest
from src.pde_solver import crank_nicolson_pricing

class TestPricingEngine(unittest.TestCase):
    def test_pde_convergence(self):
        price = crank_nicolson_pricing(S0=100, K=100, T=1.0, r=0.05, sigma=0.20)
        self.assertTrue(5.0 < price < 15.0)

if __name__ == '__main__':
    unittest.main()