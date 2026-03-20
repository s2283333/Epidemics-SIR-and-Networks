from matplotlib import pyplot as plt
import numpy as np
from basic_networks import Networks
from scipy.optimize import curve_fit


def compute_thresholds(ps, alphas, n_seeds):
    """
    Calculate epidemic threshold for a given array of ps. This allows more targetted
    alpha ranges for diffferent ranges of p, saving time. 
    
    Args:
        ps: Array of p values
        alphas: Array of alpha values to test over

    Returns:
        _type_: _description_
    """
    all_alpha_c = np.full(len(ps), np.nan)
    all_R0_c = np.full(len(ps), np.nan)

    for ip, p in enumerate(ps):
        print(f"p = {p:.3f} ({ip+1}/{len(ps)})")

        A = Networks.small_world_2d_torus_k8(L=L, p=p, seed=0)
        final_fractions = np.full((len(alphas), n_seeds), np.nan)

        # Run simulations across the alpha grid
        for j, alpha in enumerate(alphas):
            for isd in range(n_seeds):
                t, S, I, R, _, ever_inf, _ = Networks.run_sirs(
                    A,
                    alpha=float(alpha),
                    gamma=0.14,
                    omega=0.0,
                    I0=10,
                    T=1000,
                    seed=isd + 1000,
                    is_sirs=False
                )
                final_fractions[j, isd] = ever_inf[-1]

        # Compute variability and locate its peak
        mean_r = np.nanmean(final_fractions, axis=1)
        mean_r2 = np.nanmean(final_fractions**2, axis=1)
        delta = np.sqrt(np.maximum(mean_r2 - mean_r**2, 0)) / (mean_r + 1e-12)

        # Extract the relevant alpha
        alpha_c = alphas[np.argmax(delta)]

        all_alpha_c[ip] = alpha_c
        
        #R0 conversion (gamma=0.14)
        all_R0_c[ip] = alpha_c * 8 / 0.14

        print(f"  alpha_c = {alpha_c:.4f}")
        print(f"  R0_c = {all_R0_c[ip]:.3f}")

    return all_alpha_c, all_R0_c


def fit_models(p, R0_mean, R0_se, is_small_dataset=True):
    """
    Fit exponential and logarithmic models to R0(p). Returns chi-squared and residuals.
    """

    # Exponential model
    def exp_plateau(p, R_inf, A, k):
        return R_inf + A * np.exp(-k * p)
    
    # Logarithmic model (not used)
    def log_model(p, a, b):
        return a + b * np.log(p)

    
    # For fitting our p<0.05 data.
    if is_small_dataset:
        mask = p < 0.05
        p = p[mask]
        R0 = R0_mean[mask]
        R0_err = R0_se[mask]
    else:
        R0 = R0_mean
        R0_err = R0_se

    # Exponential fit
    exp_params, _ = curve_fit(
        exp_plateau,
        p,
        R0,
        sigma=R0_err,
        absolute_sigma=True,
        p0=[1.4, 1.2, 50],
        maxfev=10000
    )

    R_inf, A, k = exp_params
    exp_pred = exp_plateau(p, R_inf, A, k)
    exp_residuals = R0 - exp_pred
    chi2_exp = np.sum((exp_residuals / R0_err) ** 2)


    # Exponential diagnostic plot
    # NB. R_inf and lambda are used to mean the same thing (the expected value of R0 at p=infinity)
    R_excess = R0 - R_inf
    valid_exp = R_excess > 0

    p_exp = p[valid_exp]
    R_excess_exp = R_excess[valid_exp]
    R0_err_exp = R0_err[valid_exp]

    log_vals = np.log(R_excess_exp)
    log_se = R0_err_exp / R_excess_exp
    coeffs = np.polyfit(p_exp, log_vals, 1, w=1 / log_se)

    plt.figure()
    plt.errorbar(
        p_exp,
        log_vals,
        yerr=log_se,
        fmt='o',
        markersize=4,
        capsize=3,
        label=r'$\ln(R_0 - \lambda)$'
    )
    plt.plot(
        p_exp,
        np.polyval(coeffs, p_exp)
    )
    plt.xlabel('Rewiring probability $p$')
    plt.ylabel(r'$\ln(R_0 - \lambda)$')
    if is_small_dataset:
        plt.title('Low-$p$ Exponential Fit Test')
    else:
        plt.title('Exponential Fit over Full $p$ Range')
    plt.show()


    # Logarithmic fit
    
    # Preventing error
    mask_log = p > 0
    p_log = p[mask_log]
    R_log = R0[mask_log] - R_inf
    R_err_log = R0_err[mask_log]

    log_params, _ = curve_fit(
        log_model,
        p_log,
        R_log,
        sigma=R_err_log,
        absolute_sigma=True
    )

    a, b = log_params
    log_pred = log_model(p_log, a, b) + R_inf
    log_residuals = R0[mask_log] - log_pred
    chi2_log = np.sum((log_residuals / R0_err[mask_log]) ** 2)


    # Logarithmic fit plot
    x_log = np.log(p_log)
    x_curve = np.linspace(x_log.min(), x_log.max(), 400)
    p_curve = np.exp(x_curve)

    plt.figure()
    plt.errorbar(
        x_log,
        R_log,
        yerr=R_err_log,
        fmt='o',
        markersize=4,
        capsize=3,
        label=r'$R_0 - \lambda$'
    )
    plt.plot(
        x_curve,
        log_model(p_curve, a, b)
    )
    plt.xlabel(r'$\ln(p)$')
    plt.ylabel(r'$R_0 - \lambda$')
    plt.title('Low-$p$ Logarithmic Fit Test')
    plt.show()

    return exp_params, log_params, chi2_exp, chi2_log, coeffs, exp_residuals, log_residuals



def run_fft_over_seeds(p, n_seeds=10):
    """Function to find maximum frequency in endemic limit over multiple seeds"""
    fft_vals = []

    for seed in range(n_seeds):
        A = Networks.small_world_2d_torus_k8(L=400, p=p, seed=seed)

        t, S, I, R, _, _, _ = Networks.run_sirs(
            A,
            alpha=0.5/8,
            gamma=0.14,
            omega=0.01,
            I0=10,
            T=10000,
            seed= seed+1000 # Keep constant
        )
        vals, pers =Networks.max_fft(I)
        fft_vals.append(pers)

    return np.array(fft_vals)

def collect_peak_stats(Ls, ps, n_sims=10, T=500, alpha=0.5/8, gamma=0.14, omega=0.01, I0=10):
    """
    Collects the number infected and time thereof when the peak infection occurs at a range of ps.
    Runs over many Ls for comparison across system sizes. Repeats over multiple seeds to get a mean
    and errors.
    """
    
    # Set up dicts to have the system size (L**2) attached for each set of results.
    results = {}
    errors = {}
    peak_times = {}
    peak_time_errors = {}

    for L in Ls:
        print(f"\nStarting simulations for L={L}")

        mean_peaks = []
        std_errors = []
        mean_peak_times = []
        std_error_peak_times = []

        for p in ps:
            print(f"  p = {p}")

            peaks = []
            times = []

            for sim in range(n_sims):
                print(f"    simulation {sim+1}/{n_sims}")

                A = Networks.small_world_2d_torus_k8(L=L, p=p, seed=sim)

                t, S, I, R, new_inf, ever_inf_fraction, final_state = Networks.run_sirs(
                    A,
                    alpha=alpha,
                    gamma=gamma,
                    omega=omega,
                    I0=I0,
                    T=T,
                    seed=sim + 100,
                    is_sirs=False  # No SIRS as we just look at the first peak.
                )

                N = A.shape[0]
                peak_idx = np.argmax(I)
                peak_pct = 100 * I[peak_idx] / N
                peak_time = t[peak_idx]

                peaks.append(peak_pct)
                times.append(peak_time)

            peaks = np.array(peaks)
            times = np.array(times)

            mean_peaks.append(np.mean(peaks))
            std_errors.append(np.std(peaks, ddof=1) / np.sqrt(len(peaks)))

            mean_peak_times.append(np.mean(times))
            std_error_peak_times.append(np.std(times, ddof=1) / np.sqrt(len(times)))

        results[L] = np.array(mean_peaks)
        errors[L] = np.array(std_errors)
        peak_times[L] = np.array(mean_peak_times)
        peak_time_errors[L] = np.array(std_error_peak_times)

    return results, errors, peak_times, peak_time_errors

def cumulative_infection_plot(
        L,
        alpha,
        gamma=0.14,
        omega=0.01,
        I0=10,
        T=1000,
        ps=(0.0, 0.01, 0.05),
        seeds=range(10),
        is_sirs=False
    ):
    """
    Plots the cumulative percentage infected at diffferent ps. 
    Created as helper function as the plotting would be quite long otherwise
    """
    colour_map = {
        0.0: "tab:blue",
        0.01: "tab:orange",
        0.05: "tab:green"
    }

    plt.figure(figsize=(8, 5))

    for p in ps:
        colour = colour_map.get(p, None)

        for seed in seeds:
            A = Networks.small_world_2d_torus_k8(L=L, p=p, seed=seed)

            _, S, I, R, _, ever_inf_frac, _ = Networks.run_sirs(
                A=A,
                alpha=alpha,
                gamma=gamma,
                omega=omega,
                I0=I0,
                T=T,
                seed=seed,
                is_sirs=is_sirs
            )

            plt.plot(
                np.arange(T + 1),
                ever_inf_frac,
                color=colour,
                linewidth=1.0,
                label=f"p = {p}" if seed == 0 else None
            )

    plt.title("Cumulative Infection for Different Rewiring Probabilities")
    plt.xlabel("Time (days)")
    plt.ylabel("Cumulative Fraction Ever Infected")
    plt.legend(loc="upper left")
    plt.show()
    

def sweep_secondary_fraction(L, alpha, gamma, omega, I0=5, T=50,
                              n_p=15, p_max=0.05, n_runs=10,
                              is_sirs=False):
    """
    Sweep p from 0 to p_max and plot the fraction of infections occuring PURELY from secondary seeds.
    """
    ps = np.linspace(0, p_max, n_p)
    means = np.zeros(n_p)
    stds  = np.zeros(n_p)
    serrs = np.zeros(n_p)
    for i, p in enumerate(ps):
        vals = []

        for seed in range(n_runs):
            A = Networks.small_world_2d_torus_k8(L, p=p, seed=seed)

            *_, frac_secondary, _ = Networks.run_sirs_lineage(
                A, L, alpha, gamma, omega,
                I0=I0, T=T, seed=seed+1000, is_sirs=is_sirs
            )

            vals.append(frac_secondary[-1])

        means[i] = np.mean(vals)
        stds[i]  = np.std(vals)
        serrs[i] = stds[i]/np.sqrt(len(vals))
        print(f"p={p:.4f}  fraction = {means[i]:.3f} ± {stds[i]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.errorbar(
        ps,
        means,
        yerr=serrs,
        fmt='o-',
        capsize=4
    )

    ax.set_xlabel("Rewiring probability $p$")
    ax.set_ylabel(f"Fraction at t = {T}")
    ax.set_title("Fraction of Infections from Secondary Seeds")
    ax.set_xlim(0, p_max)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
    return ps, means, stds

