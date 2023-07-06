from typing import Any, Tuple

import numpy as np
from numpy.linalg import svd


class HierarchicalSVD:
    def __init__(self, k: int = 3, w_0: int = 5, W: int = 2, L: int = 5) -> None:
        self.k = k
        self.w_0 = w_0
        self.W = W
        self.L = L

    def set_params(self, **parameters: Any) -> "HierarchicalSVD":
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X: np.ndarray) -> None:
        pi_dict = dict()
        V0_ls = []
        V, P, _, pi = self.__local_pattern(X, self.w_0)
        V0_ls.append(V.T)
        pi_dict[self.w_0] = pi
        for l in range(1, self.L):
            w_l = self.w_0 * self.W**l
            V, P, _, pi = self.__local_pattern(P, self.W)
            V0_ls.append(np.zeros((w_l, self.k)))
            pi_dict[w_l] = pi
            for i in range(self.k):
                for j in range(self.W):
                    V0_ls[l][:, i][j * w_l // self.W : (j + 1) * w_l // self.W] = (
                        V0_ls[l - 1] @ V.T[:, i][j * self.k : (j + 1) * self.k]
                    )
        self.set_params(P=P, V=V0_ls[-1], pi_dict=pi_dict)

    def __delay(self, X: np.ndarray, w: int) -> np.ndarray:
        m = X.shape[0] // w
        n = w if len(X.shape) == 1 else X.shape[1] * w
        _X = np.zeros((m, n))
        for i in range(m):
            _X[i, :] = X[i * w : (i + 1) * w].reshape(1, -1)

        return _X

    def __local_pattern(
        self,
        X: np.ndarray,
        w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        X_w = self.__delay(X, w)
        U, S, V = svd(X_w, full_matrices=False)
        s_matrix = np.diag(S[: self.k])
        pi = sum(S[self.k :] ** 2) / w

        return V[: self.k].round(2), (U[:, : self.k] @ s_matrix).round(2), s_matrix.round(2), pi
