import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cond
from numpy.random import choice
from scipy.linalg import svd
from scipy.stats import norm
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from loguru import logger

from lib.utility import SimulateData


class MoceAlg:
    """method of contraction and expansion"""

    def __init__(self, x, y, beta_0=None):
        self.n = y.shape[0]
        self.p = x.shape[1]
        self.x = x
        self.y = y
        self.beta_0 = beta_0
        self.beta_hat = None
        self.a_hat = None
        self.dl = None
        self.a_tilde = None
        self.beta_moce = None
        self.beta_moce_astd = None
        self.sn_c_a = None
        self.tau_a = None
        self.tau_c = None
        self.inv_sigma_a = None
        self.variance = None
        self.beta_moce_ci = None
        self.l_sn_l = None
        self.r_a = None
        self.r_c = None
        self.inv_sigma_c = None
        self.beta_moce_cp = None

    @classmethod
    def moce(cls, x, y):
        cls.dat = SimulateData.add_dat(x, y)
        return cls

    def contraction(self):
        """Model contraction using lasso and bic criterion"""

        model_bic = LassoLarsCV(
            fit_intercept=False, verbose=True, normalize=False, cv=20
        ).fit(self.x, self.y)

        self.beta_hat = model_bic.coef_
        self.a_hat = abs(self.beta_hat) > 0
        self.dl = (
            -(
                self.x.transpose().dot(self.y)
                - self.x.transpose().dot(self.x).dot(self.beta_hat)
            )
            / self.n
        )

    def expansion(self, a_max_size, a_offset):
        """Model expansion using forward regression selection"""
        self.a_tilde = np.zeros(self.p, dtype=bool)
        bic = np.zeros(a_offset)
        path = np.zeros(a_offset, dtype=int)
        path[: self.a_tilde.sum()] = np.where(self.a_tilde)[0]
        j = self.a_tilde.sum()
        while self.a_tilde.sum() < a_offset:
            rss_k = np.zeros(self.p)
            for k in range(self.p):
                if self.a_tilde[k]:
                    continue
                a_curr = self.a_tilde
                a_curr[k] = True
                x_k = self.x[:, a_curr]
                h_k = x_k.dot(
                    np.linalg.inv(x_k.transpose().dot(x_k)).dot(x_k.transpose())
                )
                rss_k[k] = self.y.transpose().dot(self.y) - self.y.transpose().dot(
                    h_k
                ).dot(self.y)
                a_curr[k] = False
            min_rss_k = np.where(rss_k == rss_k[rss_k > 0].min())
            self.a_tilde[min_rss_k] = True
            path[j] = min_rss_k[0]
            bic[j] = np.log(
                rss_k[min_rss_k] / self.n
            ) + self.n**-1 * self.a_tilde.sum() * (np.log(self.n))
            j += 1

        # min_a_size = np.where(bic == bic[np.abs(bic) > 0].min())[0][0]
        self.a_tilde[self.a_hat] = True
        self.a_tilde[path[:a_offset]] = True
        a_random = a_max_size - a_offset
        if a_random > 0:
            random_set = np.random.choice(
                np.where(self.a_tilde == False)[0], a_random, replace=False
            )
            self.a_tilde[random_set] = True

    def inference(self, x, y, tau_a, tau_c, print_progress=False, delta=1e-0):
        """MOCE de-biased estimator and its inference"""
        logger.info(f"inside inference tau_c = {tau_c}")
        self.beta_moce = np.zeros(self.p)
        self.beta_moce_astd = np.zeros(self.p)

        n = x.shape[0]
        sn_a = x[:, self.a_tilde].transpose().dot(x[:, self.a_tilde]) / n
        sn_c = x[:, ~self.a_tilde].transpose().dot(x[:, ~self.a_tilde]) / n
        self.sn_c_a = x[:, ~self.a_tilde].transpose().dot(x[:, self.a_tilde]) / n

        rho_c = svd(sn_c, compute_uv=False)
        rho_a = svd(sn_a, compute_uv=False)
        self.tau_a = (
            tau_a * rho_a[-1] / (np.log(self.p) ** 0.5 * self.a_tilde.sum()) / delta
        )
        self.sigma_a = sn_a.copy()
        self.sigma_a[np.diag_indices(self.sigma_a.shape[0])] += self.tau_a
        self.inv_sigma_a = np.linalg.inv(self.sigma_a)
        self.beta_moce[self.a_tilde] = self.beta_hat[
            self.a_tilde
        ] - self.inv_sigma_a.dot(self.dl[self.a_tilde])

        # equ I_11
        # self.r_a = ((np.linalg.norm(self.tau_a * self.inv_sigma_a.dot(
        #    self.beta_hat[self.a_tilde]))) ** 2 / self.a_tilde.sum()) ** 0.5

        self.r_a = np.abs(
            -self.tau_a * self.inv_sigma_a.dot(self.beta_hat[self.a_tilde])
            + self.inv_sigma_a.dot(self.sn_c_a.transpose()).dot(
                self.beta_hat[~self.a_tilde]
            )
        ).max()
        self.tau_c = tau_c * (rho_c[0] * rho_a[0]) ** 0.5 / delta
        self.sigma_c = sn_c.copy()
        self.sigma_c[np.diag_indices(self.sigma_c.shape[0])] += self.tau_c
        self.inv_sigma_c = np.linalg.pinv(self.sigma_c)
        self.beta_moce[~self.a_tilde] = (
            self.beta_hat[~self.a_tilde]
            - self.inv_sigma_c.dot(self.dl[~self.a_tilde])
            + self.inv_sigma_c.dot(self.sn_c_a)
            .dot(self.inv_sigma_a)
            .dot(self.dl[self.a_tilde])
        )

        # equ I_21 + I_22
        self.r_c = (
            (
                np.linalg.norm(
                    self.tau_c * self.inv_sigma_c.dot(self.beta_hat[~self.a_tilde])
                )
                + np.linalg.norm(self.inv_sigma_c.dot(self.sn_c_a))
            )
            ** 2
            / (self.p - self.a_tilde.sum())
        ) ** 0.5
        self.variance = np.sum((x.dot(self.beta_hat) - y) ** 2) / (n - self.a_hat.sum())

        if print_progress:
            print(
                """lasso_std={:6.4f}, a_tilde_size={:>2}, 
                moce_true_positive={:>2}, a_hat={:>2}, 
                lasso_true_positive ={:>2}, 
                tau_a={:.4E}, 
                tau_c={:.4E}, 
                r_a={:.4E}, 
                r_c={:.4E}""".format(
                    self.variance**0.5,
                    self.a_tilde.sum(),
                    np.sum(np.abs(self.beta_0[self.a_tilde]) > 0),
                    self.a_hat.sum(),
                    np.sum(np.abs(self.beta_0[self.a_hat]) > 0),
                    self.tau_a,
                    self.tau_c,
                    self.r_a,
                    self.r_c,
                )
            )

        with np.errstate(invalid="raise"):
            self.beta_moce_astd[self.a_tilde] = (
                self.inv_sigma_a.dot(sn_a).dot(self.inv_sigma_a).diagonal()
                * self.variance
            ) ** 0.5

        mat = x[:, ~self.a_tilde].transpose() - self.sn_c_a.dot(self.inv_sigma_a).dot(
            x[:, self.a_tilde].transpose()
        )
        self.beta_moce_astd[~self.a_tilde] = (
            (
                self.inv_sigma_c.dot(mat.dot(mat.transpose()) / n).dot(self.inv_sigma_c)
            ).diagonal()
            * self.variance
        ) ** 0.5
        l = np.zeros((self.p, self.p))
        l[np.ix_(np.where(self.a_tilde)[0], np.where(self.a_tilde)[0])] = self.sigma_a
        l[
            np.ix_(
                np.where(self.a_tilde == False)[0], np.where(self.a_tilde == False)[0]
            )
        ] = self.sigma_c
        l[
            np.ix_(np.where(self.a_tilde == False)[0], np.where(self.a_tilde)[0])
        ] = self.sn_c_a
        l_inv = np.linalg.pinv(l)
        sn = self.x.transpose().dot(self.x) / self.n
        self.l_sn_l = l_inv.dot(sn).dot(l_inv.transpose())

    def get_confidence(self, sig_level):
        """Calculate confidence intervals for the moce estimator"""
        self.beta_moce_ci = np.zeros((self.p, 2))
        self.beta_moce_ci[:, 0] = (
            self.beta_moce
            - norm.ppf(1 - sig_level / 2) * self.beta_moce_astd / self.n**0.5
        )
        self.beta_moce_ci[:, 1] = (
            self.beta_moce
            + norm.ppf(1 - sig_level / 2) * self.beta_moce_astd / self.n**0.5
        )
        self.beta_moce_cp = (self.beta_moce_ci[:, 0] <= self.beta_0) & (
            self.beta_moce_ci[:, 1] >= self.beta_0
        )

    def select_tau_a_cv(self, tau_a_k, tau_c, k_fold, delta_tau, plot_fig=False):
        """Select tau_a parameter using cross validation"""
        score = np.zeros(tau_a_k.shape[0])
        r_a = np.zeros(tau_a_k.shape[0])
        for i in range(tau_a_k.shape[0]):

            k = 0
            while k < k_fold:
                x_train, x_test, y_train, y_test = train_test_split(
                    self.x, self.y, test_size=1 / k_fold, random_state=0 + k
                )
                try:

                    logger.info(f"outside inference tau_c ={tau_c}")
                    self.inference(x_train, y_train, tau_a_k[i], tau_c, False)
                    score[i] += (
                        y_test
                        - x_test[:, self.a_tilde].dot(self.beta_moce[self.a_tilde])
                    ).dot(
                        y_test
                        - x_test[:, self.a_tilde].dot(self.beta_moce[self.a_tilde])
                    ) / self.x.shape[
                        0
                    ]
                    r_a[i] += self.r_a / k_fold
                    k += 1
                except np.linalg.LinAlgError:
                    continue

        if plot_fig:
            plt.plot(np.log(score), label=r"$MSE$")
            plt.plot(np.log(r_a), label=r"$Bias$")
            plt.axhline(y=np.log(10**-2 * self.n**-0.5))
            plt.plot(
                np.abs(np.log(r_a) - np.log(delta_tau * self.n**-0.5)), label=r"$MEC$"
            )
            plt.ylabel(r"$\log$-scale")
            plt.legend()

            plt.plot(np.log(tau_a_k), r_a)

            plt.figure(figsize=(7, 5))
            plt.plot(np.log(tau_a_k), r_a, label=r"$\|r_a\|$")
            plt.ylabel(r"$||r_a||_2$", fontsize=15)
            plt.xlabel(r"$\log(\tau_a)$", fontsize=15)
            plt.tick_params(axis="both", which="major", labelsize=13)
            plt.plot(np.log(tau_a_k[15]), r_a[15], "ro")
            plt.savefig("tau_a.png")
            plt.show()

            plt.plot(np.log(tau_a_k), np.gradient(r_a, np.log(tau_a_k)))
            plt.ylabel(ylabel=r"$\frac{\partial \|r_a\|}{\partial (\log(tau_a))}$")
            plt.xlabel(r"$\log(c_a)$")
            plt.savefig("d_tau_a.png")
            plt.show()

        min_index = np.where(np.gradient(r_a, np.log(tau_a_k)) > delta_tau)[0][0]
        return tau_a_k[min_index]

    def select_tau_c_cv(self, tau_a, tau_c_k, k_fold, delta_tau, plot_fig=False):
        """Select tau_c parameter"""
        r_c = np.zeros(tau_c_k.shape[0])
        for i in range(tau_c_k.shape[0]):

            j = 0
            k = 0
            while k < k_fold:
                x_train, x_test, y_train, y_test = train_test_split(
                    self.x, self.y, test_size=1 / k_fold, random_state=j + k
                )
                try:
                    self.inference(x_train, y_train, tau_a, tau_c_k[i], False)
                    r_c[i] += self.r_c / k_fold
                    k += 1
                except np.linalg.LinAlgError:
                    j += 1
                    continue
                except RuntimeWarning:
                    j += 1
                    continue

        if plot_fig:
            plt.figure(figsize=(7, 5))
            plt.plot(np.log(tau_c_k), r_c, label=r"$\|r_c\|$")
            plt.ylabel(r"$||r_c||_2$", fontsize=15)
            plt.xlabel(r"$\log(\tau_c)$", fontsize=15)
            plt.tick_params(axis="both", which="major", labelsize=13)
            plt.plot(np.log(tau_c_k[8]), r_c[8], "ro")
            # plt.savefig("tau_c.png")
            plt.show()

            plt.plot(np.log(tau_c_k), np.gradient(r_c, np.log(tau_c_k)))
            plt.ylabel(ylabel=r"$\|r_c\|$")
            plt.xlabel(r"$\log(\tau_c)$")
            # plt.savefig("d_tau_c.png")
            plt.show()

        try:
            min_index = np.where(np.gradient(r_c, np.log(tau_c_k)) < delta_tau)[0][0]

            return tau_c_k[min_index]
        except IndexError:
            return 1e-7

    def hypothesis_test(self, g_set):
        """Conduct a hypothesis test for a set of parameters g_set"""
        l_inv = np.zeros((self.p, self.p))
        l_inv[
            np.ix_(np.where(self.a_tilde)[0], np.where(self.a_tilde)[0])
        ] = self.inv_sigma_a
        l_inv[
            np.ix_(
                np.where(self.a_tilde == False)[0], np.where(self.a_tilde == False)[0]
            )
        ] = self.inv_sigma_c
        l_inv[
            np.ix_(np.where(self.a_tilde == False)[0], np.where(self.a_tilde)[0])
        ] = -self.inv_sigma_c.dot(self.sn_c_a).dot(self.inv_sigma_a)

        sn = self.x.transpose().dot(self.x) / self.n
        self.l_sn_l = l_inv.dot(sn).dot(l_inv.transpose())
        tr_s = self.l_sn_l[np.ix_(g_set, g_set)].diagonal().sum()
        tr_s_2 = np.diagonal(
            self.l_sn_l[np.ix_(g_set, g_set)].dot(self.l_sn_l[np.ix_(g_set, g_set)])
        ).sum()

        num = (
            self.beta_moce[g_set].dot(self.beta_moce[g_set]) * self.n
            - self.variance * tr_s
        )

        deno = self.variance * (2 * tr_s_2) ** 0.5
        w_bc = num / deno
        w_1 = (
            self.n
            * self.beta_moce[g_set]
            .dot(np.linalg.inv(self.l_sn_l[np.ix_(g_set, g_set)]))
            .dot(self.beta_moce[g_set])
            / self.variance
        )
        w_2 = (
            self.n
            * np.sum(
                self.beta_moce[g_set] ** 2 / np.diag(self.l_sn_l[np.ix_(g_set, g_set)])
            )
            / self.variance
        )
        return w_bc, w_1, w_2

    def residual_bootstrap(self, x, y, b, method="residual"):
        """ "Conduct residual bootstrap with the bootstrap replication b"""
        self.beta_moce_b = np.zeros((self.a_tilde.sum(), b))
        if method == "residual":
            self.residual_a = y - x[:, self.a_tilde].dot(self.beta_hat[self.a_tilde])

            for i in range(b):
                y_b = x[:, self.a_tilde].dot(self.beta_moce[self.a_tilde]) + choice(
                    self.residual_a, self.n
                )

                (
                    self.beta_moce_b[
                        i,
                    ],
                    _,
                    _,
                    _,
                ) = np.linalg.lstsq(x[:, self.a_tilde], y_b, rcond=None)
        elif method == "nonparametric":
            for i in range(b):
                index_i = choice(np.array(range(self.n)), self.n)
                y_b = y[index_i]
                x_b = x[
                    index_i,
                ]
                self.beta_moce_b[:, i], _, _, _ = np.linalg.lstsq(
                    x_b[:, self.a_tilde], y_b, rcond=None
                )

        self.beta_moce_bc = self.beta_moce.copy()
        self.beta_moce_bc_astd = self.beta_moce_astd.copy()
        self.beta_moce_bc[self.a_tilde] = self.beta_moce_bc[
            self.a_tilde
        ] * 2 - self.beta_moce_b.mean(axis=1)
        self.beta_moce_bc_astd[self.a_tilde] = self.beta_moce_b.std(axis=1)


class Bootstrap:
    """
    Class for bootstrap variance
    """

    def __init__(
        self, x, y, b, min_tau_a, min_tau_c, a_max_size, a_offset, beta_0=None
    ):
        self.x = x
        self.y = y
        self.b = b
        self.n = y.shape[0]
        self.min_tau_a = min_tau_a
        self.min_tau_c = min_tau_c
        self.a_max_size = a_max_size
        self.a_offset = a_offset
        self.beta_0 = beta_0
        self.beta_moce_b = np.zeros((self.x.shape[1], b))
        self.beta_moce_bc = None
        self.beta_moce_bc_astd = None
        self.beta_moce_bc_quantile_025 = None
        self.beta_moce_bc_quantile_975 = None
        self.beta_moce_bc_quantile_005 = None
        self.beta_moce_bc_quantile_995 = None

    def moce_bootstrap(self):
        for i in range(self.b):
            index_i = choice(np.array(range(self.n)), self.n)
            y_b = self.y[index_i]
            x_b = self.x[
                index_i,
            ]

            moce_obj = MoceAlg(x_b, y_b)
            moce_obj.contraction()
            moce_obj.expansion(self.a_max_size, self.a_offset)
            moce_obj.inference(x_b, y_b, self.min_tau_a, self.min_tau_c)

            self.beta_moce_b[:, i] = moce_obj.beta_moce

        self.beta_moce_bc = self.beta_moce_b.mean(axis=1)
        self.beta_moce_bc_astd = self.beta_moce_b.std(axis=1) * self.y.shape[0] ** 0.5
        self.beta_moce_bc_quantile_025 = np.quantile(self.beta_moce_b, 0.025, axis=1)
        self.beta_moce_bc_quantile_975 = np.quantile(self.beta_moce_b, 0.975, axis=1)
        self.beta_moce_bc_quantile_005 = np.quantile(self.beta_moce_b, 0.005, axis=1)
        self.beta_moce_bc_quantile_995 = np.quantile(self.beta_moce_b, 0.995, axis=1)
