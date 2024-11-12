import numpy as np
import scipy.stats as si

class GreeksCalculator:
    def __init__(self, option_type, S, K, T, r, sigma):
        self.option_type = option_type
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def calculate_greeks(self):
        d1 = self.d1()
        d2 = self.d2()
        delta = si.norm.cdf(d1) if self.option_type == "call" else si.norm.cdf(d1) - 1
        gamma = si.norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        theta = -(self.S * si.norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2) if self.option_type == "call" else -(self.S * si.norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2)
        vega = self.S * si.norm.pdf(d1) * np.sqrt(self.T)
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
