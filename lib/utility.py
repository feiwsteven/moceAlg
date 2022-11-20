import numpy as np
from numpy.random import normal, multivariate_normal
from scipy.stats import norm, chi2


class SimulateData:
    """simulate data"""

    @classmethod
    def add_dat(cls, x, y):
        cls.x = x
        cls.y = y

    def __init__(self, n: int, p: int, beta_0: np.array, std_0: float, cov: np.array):
        self.n = n
        self.p = p
        self.cov = cov
        self.beta_0 = beta_0
        self.std_0 = std_0
        self.fixed_x = multivariate_normal(np.zeros(self.p), self.cov, self.n)
        self.fixed_x = self.fixed_x / np.sum(self.fixed_x**2 / self.n, axis=0) ** 0.5
        self.x = None
        self.y = None

    def add_cv_data(self, x, y):
        self.x = x
        self.y = y

    def simulate_linear(self, fixed_design=False):
        """Simulate a linear model"""
        if fixed_design:
            self.x = self.fixed_x
        else:
            self.x = multivariate_normal(np.zeros(self.p), self.cov, self.n)
            self.x = self.x / np.sum(self.x**2 / self.n, axis=0) ** 0.5
        self.y = self.x.dot(self.beta_0) + normal(scale=self.std_0, size=self.n)
        self.r_2 = 1 / (
            1
            + self.std_0**2
            / (
                self.n
                / np.sum(self.x**2, axis=0).mean()
                * self.beta_0.dot(self.cov).dot(self.beta_0)
            )
        )

    def add_result(
        self,
        beta_hat,
        beta_moce,
        beta_moce_astd,
        a_tilde,
        r_a,
        variance,
        beta_moce_bc=None,
        beta_moce_bc_astd=None,
        beta_moce_bc_quantile_025=None,
        beta_moce_bc_quantile_975=None,
        beta_moce_bc_quantile_005=None,
        beta_moce_bc_quantile_995=None,
    ):
        """Add simulate results"""
        self.beta_hat = beta_hat
        self.beta_moce = beta_moce
        self.beta_moce_astd = beta_moce_astd
        self.a_tilde = a_tilde
        self.r_a = r_a
        self.variance = variance
        self.beta_moce_bc = beta_moce_bc
        self.beta_moce_bc_astd = beta_moce_bc_astd
        self.beta_moce_bc_quantile_025 = beta_moce_bc_quantile_025
        self.beta_moce_bc_quantile_975 = beta_moce_bc_quantile_975
        self.beta_moce_bc_quantile_005 = beta_moce_bc_quantile_005
        self.beta_moce_bc_quantile_995 = beta_moce_bc_quantile_995

    def add_power(self, w_bc, w_1, g_size):
        """Add power results"""
        self.w_bc = w_bc
        self.w_1 = w_1
        self.g_size = g_size

    def summary(self, sig_level, moce=True):
        """Calculate summary statistics"""
        self.sig_level = sig_level
        self.beta_moce_bias = self.beta_moce.mean(axis=1) - self.beta_0
        self.beta_moce_estd = self.beta_moce.var(axis=1) ** 0.5

        ci_count = (
            self.beta_0.reshape((-1, 1))
            <= self.beta_moce
            + norm.ppf(1 - self.sig_level / 2) * self.beta_moce_astd / self.n**0.5
        ) & (
            self.beta_0.reshape((-1, 1))
            >= self.beta_moce
            - norm.ppf(1 - self.sig_level / 2) * self.beta_moce_astd / self.n**0.5
        )

        up = (
            self.beta_moce
            + norm.ppf(1 - self.sig_level / 2) * self.beta_moce_astd / self.n**0.5
        )
        low = (
            self.beta_moce
            - norm.ppf(1 - self.sig_level / 2) * self.beta_moce_astd / self.n**0.5
        )
        err_index = self.beta_0.reshape((-1, 1))

        self.acp = ci_count.sum(axis=1) / ci_count.shape[1]

        if moce:
            temp_1 = ci_count & (self.a_tilde == 1)
            temp_0 = ci_count & (self.a_tilde == 0)
            self.acp_1 = temp_1.sum(axis=1) / self.a_tilde.sum(axis=1)
            self.acp_0 = temp_0.sum(axis=1) / (
                self.a_tilde.shape[1] - self.a_tilde.sum(axis=1)
            )

        ci_count = (
            self.beta_0.reshape((-1, 1))
            <= self.beta_moce
            + norm.ppf(1 - self.sig_level / 2) * self.beta_moce_estd.reshape((-1, 1))
        ) & (
            self.beta_0.reshape((-1, 1))
            >= self.beta_moce
            - norm.ppf(1 - self.sig_level / 2) * self.beta_moce_estd.reshape((-1, 1))
        )

        self.ecp = ci_count.sum(axis=1) / ci_count.shape[1]

        if self.beta_moce_bc is not None:
            if self.sig_level == 0.05:
                low = self.beta_moce_bc_quantile_025
                up = self.beta_moce_bc_quantile_975
            elif self.sig_level == 0.01:
                low = self.beta_moce_bc_quantile_005
                up = self.beta_moce_bc_quantile_995

            ci_count = (self.beta_0.reshape((-1, 1)) <= up) & (
                self.beta_0.reshape((-1, 1)) >= low
            )

            self.acp_bc = ci_count.sum(axis=1) / ci_count.shape[1]
            self.beta_moce_bc_bias = self.beta_moce_bc.mean(axis=1) - self.beta_0

    def summary_power(self, sig_level):
        """Calculate power of hypotheses"""
        self.w_bc_power = (
            np.sum(np.abs(self.w_bc) > norm.ppf(1 - sig_level / 2), axis=1)
            / self.w_bc.shape[1]
        )
        self.w_1_power = (
            np.sum(self.w_1 > chi2.ppf(1 - sig_level, self.g_size), axis=1)
            / self.w_1.shape[1]
        )
