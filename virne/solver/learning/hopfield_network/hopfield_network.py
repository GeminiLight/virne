# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import numpy as np


class HopfieldNetwork:

    def __init__(self, chi, psi, zeta):
        self.alpha = 2.5
        self.iter_max = 2000
        self.tau = 0.02
        self._init(chi, psi, zeta)

    def _init(self, chi, psi, zeta):
        t_const = np.ones((len(chi), len(chi)))
        for i in range(zeta):
            t_const[i, i] = 0
        i_const = np.ones(len(chi)) * - (2 * zeta - 1)
        T = -2 * (psi + self.alpha * t_const)
        I = - (chi + self.alpha * i_const)

        self.T = T
        self.I = I

    def execute(self):
        u0 = 1.0
        Delta = np.inf
        delta = 1.0
        i = 0
        U = np.random.uniform(0, 1, self.I.shape)
        V = np.copy(U)
        while (Delta > delta) and (i < self.iter_max):
            i = i + 1
            k1 = np.matmul(self.T ,V) + self.I - U
            k2_temp = np.divide((U + 0.5 * self.tau * k1), u0)
            k2 = np.matmul(self.T, (0.5 * (1+np.tanh(k2_temp)))) + self.I - (U + 0.5 * self.tau *k1)
            k3_temp = np.divide((U - self.tau * k1 + 2 * self.tau * k2), u0)
            k3 = np.matmul(self.T, (0.5 * (1+np.tanh(k3_temp)))) + self.I - (U - self.tau * k1 + 2 * self.tau * k2)
            dU = (k1 + 4 * k2 + k3) / 6
            U = U + self.tau * dU
            Delta = np.sqrt(sum(np.square(dU)))
            V = 0.5 * (1 + np.tanh(np.divide(U, u0)))
        return V