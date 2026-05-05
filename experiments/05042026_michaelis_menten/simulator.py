import numpy as np
from scipy.integrate import solve_ivp


def simulate(
    alpha: float,
    gamma: float,
    KM1: float,
    KM2: float,
    KM3: float,
    k2_1: float,
    k2_2: float,
    k2_3: float,
    g1T0: float,
    g2T0: float,
    g3T0: float,
    mu: float,
    beta: float,
    Tmax: float,
    n_time_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Michaelis-Menten ODE dynamics for metabolites A, B, C, D.

    Returns A, B, C, D, g2T, t as numpy arrays of shape (n_time_points,).
    """
    Vmax1 = k2_1 * g1T0
    Vmax2_0 = k2_2 * g2T0
    Vmax3 = k2_3 * g3T0
    g2T_star = beta / mu  # steady-state enzyme level

    # Initial conditions at original steady state (g2T = g2T0)
    A0 = alpha * KM1 / (Vmax1 - alpha)
    B0 = alpha * KM2 / (Vmax2_0 - alpha)
    C0 = alpha * KM3 / (Vmax3 - alpha)
    D0 = alpha / gamma

    def vmax2(t):
        return k2_2 * (g2T_star + (g2T0 - g2T_star) * np.exp(-mu * t))

    def odes(t, y):
        A, B, C, D = y
        v1 = Vmax1 * A / (KM1 + A)
        v2 = vmax2(t) * B / (KM2 + B)
        v3 = Vmax3 * C / (KM3 + C)
        return [
            alpha - v1,
            v1 - v2,
            v2 - v3,
            v3 - gamma * D,
        ]

    t_eval = np.linspace(0, Tmax, n_time_points)
    sol = solve_ivp(
        odes,
        t_span=(0, Tmax),
        y0=[A0, B0, C0, D0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    A, B, C, D = sol.y
    g2T = g2T_star + (g2T0 - g2T_star) * np.exp(-mu * sol.t)
    return A, B, C, D, g2T, sol.t
