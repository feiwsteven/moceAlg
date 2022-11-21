import unittest
import numpy as np
from lib.moce import MoceAlg
from numpy.random import uniform
from lib.utility import SimulateData
from loguru import logger
import matplotlib.pyplot as plt


class TestMoce(unittest.TestCase):
    def test_linear_one_case(self):
        seed = 504
        alpha = 0.5
        p = 400
        n = 300
        a = 3
        a_max_size = a * 1 + int(0.05 * 1 * p)
        a_offset = a * 2
        """vary p to 200, 400, 800, and 1000"""

        n_iter = 500
        # std_0 = 2 * (a / n) ** 0.5
        std_0 = 0.5
        delta_tau_a = 1e-4
        delta_tau_c = -1e-5

        select_by_mse = False

        cov = np.identity(p)
        for i in range(p - 1):
            cov[i, i:p] = alpha ** np.arange(0, p - i)
            cov[i:p, i] = alpha ** np.arange(0, p - i)
        cov = cov * 0.25

        beta_0 = np.concatenate(
            [np.array(uniform(0.10, 0.50, size=a)), np.zeros(p - a)], axis=0
        )

        dat_obj = SimulateData(n, p, beta_0, std_0, cov)

        dat_obj.simulate_linear(True)
        moce_obj = MoceAlg(dat_obj.x, dat_obj.y, beta_0)
        moce_obj.contraction()

        moce_obj.expansion(a_max_size, a_offset)

        tau_a_k = 10 ** np.linspace(-12, 2, 20)
        tau_c_k = 10 ** np.linspace(-12, 4, 20)

        opt_tau_a = moce_obj.select_tau_a_cv(tau_a_k, 1e-1, 10, delta_tau_a)

        opt_tau_c = moce_obj.select_tau_c_cv(opt_tau_a, tau_c_k, 10, delta_tau_c)

        moce_obj.inference(dat_obj.x, dat_obj.y, opt_tau_a, opt_tau_c)
        moce_obj.get_confidence(0.05)

        print(moce_obj.beta_moce)
        print(moce_obj.beta_moce_ci)

    def test_linear_simulation(self):
        seed = 504
        alpha = 0.5
        p = 400
        n = 300
        a = 3
        a_max_size = a * 1 + int(0.05 * 1 * p)
        a_offset = a * 2
        """vary p to 200, 400, 800, and 1000"""

        n_iter = 100
        std_0 = 0.5
        delta_tau_a = 1e-4
        delta_tau_c = -1e-5

        cov = np.identity(p)
        for i in range(p - 1):
            cov[i, i:p] = alpha ** np.arange(0, p - i)
            cov[i:p, i] = alpha ** np.arange(0, p - i)
        cov = cov * 0.25

        beta_0 = np.concatenate(
            [np.array(uniform(0.10, 0.50, size=a)), np.zeros(p - a)], axis=0
        )

        dat_obj = SimulateData(n, p, beta_0, std_0, cov)

        dat_obj.simulate_linear(True)
        moce_obj = MoceAlg(dat_obj.x, dat_obj.y, beta_0)
        moce_obj.contraction()

        moce_obj.expansion(a_max_size, a_offset)
        tau_a_k = 10 ** np.linspace(-12, 2, 20)
        tau_c_k = 10 ** np.linspace(-12, 4, 20)

        opt_tau_a = moce_obj.select_tau_a_cv(tau_a_k, 1e-1, 10, delta_tau_a)
        opt_tau_c = moce_obj.select_tau_c_cv(
            opt_tau_a, tau_c_k, 10, delta_tau_c
        )

        beta_moce = np.zeros((p, n_iter))
        beta_moce_low = np.zeros((p, n_iter))
        beta_moce_up = np.zeros((p, n_iter))
        beta_hat = np.zeros((p, n_iter))
        for i in range(n_iter):
            logger.info(f"i = {i}")

            dat_obj.simulate_linear(True)
            moce_obj = MoceAlg(dat_obj.x, dat_obj.y, beta_0)
            moce_obj.contraction()

            moce_obj.expansion(a_max_size, a_offset)

            moce_obj.inference(dat_obj.x, dat_obj.y, opt_tau_a, opt_tau_c)
            moce_obj.get_confidence(0.05)
            beta_moce[:, i] = moce_obj.beta_moce
            beta_moce_low[:, i] = moce_obj.beta_moce_ci[:, 0]
            beta_moce_up[:, i] = moce_obj.beta_moce_ci[:, 1]
            beta_hat[:, i] = moce_obj.beta_hat

        print(beta_moce)


if __name__ == "__main__":
    unittest.main()
