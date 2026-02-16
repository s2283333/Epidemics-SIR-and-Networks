from matplotlib import pyplot as plt
import numpy as np
from basic_networks import Networks, run_fft_over_seeds
from standard_SIR import TwoStrainSIRSAsym, sweep_beta2_threshold
from covid_data import CovidData
from scipy.optimize import curve_fit


import numpy as np

def frac_connections_nonlocal(A, L):
    """
    Returns fraction of directed connections (stubs) that are nonlocal
    relative to Moore neighbourhood on an LxL torus.

    If this returns 0.23 -> interpret as "23% of people's connections are rewired".
    """
    A = A.tocsr()
    N = L * L
    if A.shape != (N, N):
        raise ValueError("A shape doesn't match L*L")

    I, J = A.nonzero()  # directed edges (i->j)
    xi, yi = np.divmod(I, L)
    xj, yj = np.divmod(J, L)

    dx = xj - xi
    dy = yj - yi

    # wrap to shortest displacement on torus
    dx = np.where(dx >  L/2, dx - L, dx)
    dx = np.where(dx < -L/2, dx + L, dx)
    dy = np.where(dy >  L/2, dy - L, dy)
    dy = np.where(dy < -L/2, dy + L, dy)

    local = (np.abs(dx) <= 1) & (np.abs(dy) <= 1) & ~((dx == 0) & (dy == 0))

    # fraction of stubs that are nonlocal
    return 1.0 - local.mean()

def main():

    # Covid Data
    CovidData(sheet_name='1a').plot()

    R_eff, beta = CovidData().estimate_beta_per_step(
        gamma=1/7,
        dt=7.0,      # weekly data
    )
    print(beta)
    #SIRS
    # Uses 40,000 people to compare to network, ignoring I2 for now
    m = TwoStrainSIRSAsym(
        N=40_000,
        I10=10,
        I20=0,
        beta1=0.4,
        beta2=0, # <-- set this
        gamma1=0.14,
        gamma2=1/5,
        omega1=1/100,
        omega2=0,
        t_seed=70,
        t_max=10000,
        dt=0.1,
        R10=0.0,
        R20=0.0,
        is_R1_to_I2=False,
        interventions=[]
        )
    
    t, S, I1, I2, R1, R2 = m.run()
    m.plot()
    Networks.plot_fft(I1)
    # SIRS

    # Networks

    A = Networks.small_world_2d_torus_k8(L=200, p=1, seed=1)



    Networks.animate_sirs_grid(A, L=200, beta=0.06, gamma=0.2, omega=0.01, I0=10, T=400, seed=20)
    t, S, I, R, _, _, final_state = Networks.run_sirs(A, beta=0.4/8, gamma=0.14, omega=0.01, I0=10, T=10000, seed=20)
    Networks.plot_sirs_time_series(t, S, I, R, L=200)
    Networks.plot_fft(I)

    max_Is = []
    ps = np.linspace(0, 0.2, 15)

    for p in ps:
        A = Networks.small_world_2d_torus_k8(L=200, p=p, seed=1)
        t, S, I, R, new_inf, ever_mask, final_state = Networks.run_sirs(
            A,
            beta=0.4/8,
            gamma=0.14,
            omega=0.02,
            I0=10,
            T=10000,
            seed=20,
            is_sirs=False
        )

        N = A.shape[0]
        peak_pct = 100 * np.max(I) / N
        max_Is.append(peak_pct)

    plt.plot(ps, max_Is)
    plt.xlabel("rewiring probability p")
    plt.ylabel("peak % infected")
    plt.title("Peak fraction infected vs rewiring probability")
    plt.show()
    
    # looking at how many required to be infected for I1 to die (no interventions)
    sweep_beta2_threshold()

    # Interventions
    sweep_beta2_threshold(interventions=[(10,30,0.2,0.2),(50,60,0.2,0.2)])


if __name__ == "__main__":
    main()

ps = np.linspace(0.0, 1.0, 50)
betas = np.linspace(0.017, 0.05, 30)

beta_thresholds = []
last_deads = []
first_alives = []

for i, p in enumerate(ps):
    print(f"\n---- p = {p:.3f}  ({i+1}/{len(ps)}) ----")

    A = Networks.small_world_2d_torus_k8(L=200, p=p, seed=1)

    last_dead = None
    first_alive = None

    for j, beta in enumerate(betas):
        print(f"   beta {j+1}/{len(betas)} = {beta:.5f}", end="\r")

        t, S, I, R, _, _, _ = Networks.run_sirs(
            A,
            beta=beta,
            gamma=0.14,
            omega=0.01,
            I0=10,
            T=3000,
            seed=0,
            is_sirs=True
        )

        if I[2000] == 0:
            last_dead = beta
        else:
            first_alive = beta
            break

    if last_dead is not None and first_alive is not None:
        beta_c = 0.5 * (last_dead + first_alive)
        print(f"   → beta_c ≈ {beta_c:.6f}")
    else:
        beta_c = np.nan
        print("   → threshold not bracketed")

    beta_thresholds.append(beta_c)
    last_deads.append(last_dead)
    first_alives.append(first_alive)

# convert to arrays for later use
ps = np.array(ps)
beta_thresholds = np.array(beta_thresholds)
beta_c_mean = beta_thresholds
beta_c_std  = np.full_like(beta_c_mean, 1e-6)  # tiny weights to avoid divide-by-zero

# --- p=0 value (force fit through first point) ---
i0 = np.argmin(np.abs(ps - 0.0))
beta0 = float(beta_c_mean[i0])

# --- fit data excluding p=0 (since it's enforced) ---
mask = np.isfinite(beta_c_mean) & (ps > 0)
x = ps[mask]
y = beta_c_mean[mask]
s = beta_c_std[mask]

def f_forced(p, b_inf, p0, a):
    return b_inf + (beta0 - b_inf) * np.exp(- (p / p0)**a)

# initial guesses
b_inf_guess = np.nanmedian(y[x > 0.6])
p0_guess = 0.05
a_guess = 0.7

popt, pcov = curve_fit(
    f_forced, x, y,
    p0=[b_inf_guess, p0_guess, a_guess],
    sigma=s,
    absolute_sigma=True,
    bounds=([0.0, 1e-6, 0.05], [1.0, 5.0, 5.0]),
    maxfev=20000
)

b_inf, p0, a = popt
print("Fit params:", "beta0=", beta0, "b_inf=", b_inf, "p0=", p0, "a=", a)
last_deads = np.array(last_deads)
first_alives = np.array(first_alives)

# target
R0_target = 2.6
gamma = 0.14
k_mean = 8.0

# convert to alpha target
alpha_target = R0_target * gamma / k_mean
print("alpha_target =", alpha_target)

# solve b_inf + (beta0-b_inf)*exp(-(p/p0)^a) = alpha_target
ratio = (alpha_target - b_inf) / (beta0 - b_inf)
print("ratio =", ratio)

if not (0 < ratio < 1):
    print("No real solution: target is outside the fitted curve range.")
else:
    p_at_R0 = p0 * (-np.log(ratio))**(1.0 / a)
    print("p where R0=2.6 (from smooth fit) =", p_at_R0)

    # quick sanity check
    alpha_check = b_inf + (beta0 - b_inf) * np.exp(- (p_at_R0 / p0)**a)
    R0_check = alpha_check * k_mean / gamma
    print("check R0 =", R0_check)
    
A = Networks.small_world_2d_torus_k8(L=400, p=0, seed=0)



t, S, I, R, _, _, final_state = Networks.run_sirs(A, beta=0.6/8, gamma=0.14, omega=0.01, I0=5, T=1000, seed=0)
anim = Networks.snapshot_sirs_grid(A,400, 0.6/8,0.14,0.01, t_freeze=250, seed=0, I0=5)
Networks.plot_sirs_time_series(t, S, I, R,L=400, vline=250)

A = Networks.small_world_2d_torus_k8(L=400, p=0.01, seed=0)



t, S, I, R, _, _, final_state = Networks.run_sirs(A, beta=0.6/8, gamma=0.14, omega=0.01, I0=5, T=1000, seed=0)
anim = Networks.snapshot_sirs_grid(A,400, 0.6/8,0.14,0.01, t_freeze=250, seed=0, I0=5)
Networks.plot_sirs_time_series(t, S, I, R,L=400, vline=250)

A = Networks.small_world_2d_torus_k8(L=400, p=0.02, seed=0)



t, S, I, R, _, _, final_state = Networks.run_sirs(A, beta=0.6/8, gamma=0.14, omega=0.01, I0=5, T=1000, seed=0)
anim = Networks.snapshot_sirs_grid(A,400, 0.6/8,0.14,0.01, t_freeze=250, seed=0, I0=5)
Networks.plot_sirs_time_series(t, S, I, R,L=400, vline=250)



def count_hubs_before_peak(A, L, beta, gamma, omega, I0=10, T=10000, seed=0):
    """
    "Hub created" (as before): a new infection at time t that has NO infected
    neighbour among its 8 LOCAL grid neighbours at time t-1 (i.e. seeded via a shortcut).

    Returns:
        hubs_before_peak : int
        t_peak           : int
        I_series         : (T+1,) int array
    """
    rng = np.random.default_rng(seed)
    N = L * L

    # --- local 8-neighbour list (torus) ---
    local_nbrs = [[] for _ in range(N)]
    for x in range(L):
        for y in range(L):
            i = x * L + y
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx = (x + dx) % L
                    yy = (y + dy) % L
                    local_nbrs[i].append(xx * L + yy)

    # --- init state: 0=S, 1=I, 2=R ---
    state = np.zeros(N, dtype=np.int8)
    init = rng.choice(N, size=min(I0, N), replace=False)
    state[init] = 1

    I_series = np.zeros(T + 1, dtype=int)
    I_series[0] = np.sum(state == 1)

    hubs_per_t = np.zeros(T + 1, dtype=int)

    for t in range(1, T + 1):
        prev_state = state.copy()

        infected_prev = (prev_state == 1)
        susceptible_prev = (prev_state == 0)
        recovered_prev = (prev_state == 2)

        # k_inf for each node (number of infected neighbours in A)
        k_inf = A.dot(infected_prev.astype(int))

        # S -> I with prob 1 - (1-beta)^k_inf
        p_inf = 1.0 - np.power(1.0 - beta, k_inf)
        new_inf = susceptible_prev & (rng.random(N) < p_inf)

        # I -> R with prob gamma
        new_rec = infected_prev & (rng.random(N) < gamma)

        # R -> S with prob omega
        new_sus = recovered_prev & (rng.random(N) < omega)

        # apply updates (sync)
        state[new_inf] = 1
        state[new_rec] = 2
        state[new_sus] = 0

        I_series[t] = np.sum(state == 1)

        # count "hubs created" at this t: new infection with zero LOCAL infected neighbours at t-1
        if np.any(new_inf):
            idx = np.flatnonzero(new_inf)
            c = 0
            for i in idx:
                # any local neighbour infected at t-1?
                if not np.any(infected_prev[local_nbrs[i]]):
                    c += 1
            hubs_per_t[t] = c

    t_peak = int(np.argmax(I_series))
    hubs_before_peak = int(np.sum(hubs_per_t[:t_peak]))  # strictly before peak time

    return hubs_before_peak, t_peak, I_series


# example usage:
# A = Networks.small_world_2d_torus_k8(L=400, p=0.03, seed=11)
# hubs, tpk, I = count_hubs_before_peak(A, L=400, beta=0.4/8, gamma=0.14, omega=0.02, I0=10, T=10000, seed=20)
# print("hubs_before_peak =", hubs, "t_peak =", tpk)


def plot_cum_hubs_vs_t_over_p(L, beta, gamma, omega,
                             I0=10, T=500,
                             ps=None, n_seeds=10,
                             base_seed=0):
    """
    For each p in ps:
      - run n_seeds simulations (seed = base_seed + s)
      - compute cumulative "hubs created" curve H(t)=sum_{u<=t} hubs_per_u
      - plot mean H(t) with +/- standard error band

    "Hub created" definition (as before):
      new infection at time t with NO infected neighbour among its 8 LOCAL grid neighbours at time t-1.
    """
    if ps is None:
        ps = np.linspace(0.0, 0.1, 10)

    N = L * L

    # --- local 8-neighbour list (torus) ---
    local_nbrs = [[] for _ in range(N)]
    for x in range(L):
        for y in range(L):
            i = x * L + y
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx = (x + dx) % L
                    yy = (y + dy) % L
                    local_nbrs[i].append(xx * L + yy)

    def cum_hubs_curve_for_seed(A, seed):
        rng = np.random.default_rng(seed)

        state = np.zeros(N, dtype=np.int8)  # 0=S,1=I,2=R
        init = rng.choice(N, size=min(I0, N), replace=False)
        state[init] = 1

        hubs_per_t = np.zeros(T + 1, dtype=float)
        # hubs_per_t[0]=0

        for t in range(1, T + 1):
            prev_state = state.copy()

            infected_prev = (prev_state == 1)
            susceptible_prev = (prev_state == 0)
            recovered_prev = (prev_state == 2)

            # infections
            k_inf = A.dot(infected_prev.astype(int))
            p_inf = 1.0 - np.power(1.0 - beta, k_inf)
            new_inf = susceptible_prev & (rng.random(N) < p_inf)

            # count hubs at this t (based on prev infected among LOCAL neighbours)
            if np.any(new_inf):
                idx = np.flatnonzero(new_inf)
                c = 0
                for i in idx:
                    if not np.any(infected_prev[local_nbrs[i]]):
                        c += 1
                hubs_per_t[t] = c

            # recover / wane
            new_rec = infected_prev & (rng.random(N) < gamma)
            new_sus = recovered_prev & (rng.random(N) < omega)

            # apply updates
            state[new_inf] = 1
            state[new_rec] = 2
            state[new_sus] = 0

        return np.cumsum(hubs_per_t)  # (T+1,)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(T + 1)

    for p in ps:
        curves = np.zeros((n_seeds, T + 1), dtype=float)

        for s in range(n_seeds):
            seed = base_seed + s

            A = Networks.small_world_2d_torus_k8(L=L, p=float(p), seed=seed)
            curves[s] = cum_hubs_curve_for_seed(A, seed=seed)

        mean = curves.mean(axis=0)
        se = curves.std(axis=0, ddof=1) / np.sqrt(n_seeds)

        ax.plot(t, mean, label=f"p={p:.3f}")
        ax.fill_between(t, mean - se, mean + se, alpha=0.2)

    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative hubs created")
    ax.set_title(f"Cumulative hubs vs time (T={T}), mean ± SE over {n_seeds} seeds")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()