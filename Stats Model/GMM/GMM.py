import numpy as np
import scipy.stats as st


class GMM:
    def __init__(self, X, k=2):
        X = np.asarray(X)
        self.m, self.n = X.shape
        self.data = X.copy()
        self.k = k

    def _intitialize_parameters(self):
        self.mu_arr = np.asmatrix(np.random.random((self.k, self.n)) + np.mean(self.data))
        self.Sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.p_arr = np.ones(self.k) / self.k
        self.omega_matrix = np.asmatrix(np.empty((self.m, self.k)), dtype=float)

    def _expectation_step(self):
        for j in range(self.m):
            sum_density = 0
            for i in range(self.k):
                multi_normal_density = st.multivariate_normal.pdf(self.data[j, :], self.mu_arr[i].A1,
                                                                  self.Sigma_arr[i]) * self.p_arr[i]
                sum_density += multi_normal_density
                # print(multi_normal_density)
                self.omega_matrix[j, i] = multi_normal_density

            const_p_x_theta = 1 / sum_density

            self.omega_matrix[j, :] = self.omega_matrix[j, :] * const_p_x_theta
            assert self.omega_matrix[j, :].sum() - 1 < 1e-4

    def _maximaization_step(self):
        for i in range(self.k):
            sum_omega_for_ith_norm = self.omega_matrix[:, i].sum()
            self.p_arr[i] = 1 / self.m * sum_omega_for_ith_norm
            mu = np.zeros(self.n)
            Sigma = np.zeros((self.n, self.n))

            for j in range(self.m):
                mu += self.omega_matrix[j, i] * self.data[j, :]
                dis = (self.data[j, :] - self.mu_arr[i, :]).T * (self.data[j, :] - self.mu_arr[i, :])
                Sigma += self.omega_matrix[j, i] * dis

            self.mu_arr[i] = mu / sum_omega_for_ith_norm
            self.Sigma_arr[i] = Sigma / sum_omega_for_ith_norm

    def _calc_loglikelihood(self):
        res = 0

        for j in range(self.m):
            prob_of_occurrence_of_point_j = 0
            for i in range(self.k):
                prob_of_occurrence_of_point_j += \
                    self.p_arr[i] * st.multivariate_normal.pdf(self.data[j, :], self.mu_arr[i, :].A1,
                                                               self.Sigma_arr[i, :])

            res += np.log(prob_of_occurrence_of_point_j)

        return res

    def fit(self, epsilon=1e-4):
        self._intitialize_parameters()
        num_iters = 0
        new_log_likelihood = 1
        prev_log_likelihood = 0

        while (new_log_likelihood - prev_log_likelihood > epsilon):
            prev_log_likelihood = self._calc_loglikelihood()
            self._expectation_step()
            self._maximaization_step()
            num_iters += 1
            new_log_likelihood = self._calc_loglikelihood()
            print(f'Loop {num_iters}: Log-likelihood value: {new_log_likelihood}')

        print(f'End at {num_iters}-th loop: Log-likelihood value: {new_log_likelihood}')











