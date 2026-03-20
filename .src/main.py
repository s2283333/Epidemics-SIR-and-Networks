"""
Main script demonstrating example usage of the report's core components. Combines the
Networks class and helper_functions module to expose all required functionality.

Sections

Visuals:
    Generates snapshot animations and time series plots for small-world networks at
    varying rewiring probabilities (p = 0.00, 0.01, 0.05) to illustrate how rewiring
    affects epidemic spread across the grid.

Epidemic Threshold:
    Sweeps over rewiring probabilities and transmission rates to estimate the critical
    threshold alpha_c (and corresponding R0_c) at which an epidemic takes hold. Run
    across multiple network realisations and seeds, then fitted with exponential and
    logarithmic models. Residuals are plotted for both fits. A reduced dataset focused
    on small p values is also analysed separately. This takes a particularly long time to run.

Peak Statistics:
    Collects the peak infection percentage and time of peak infection as a function of
    rewiring probability, for grid sizes L = 200, 300, and 400.

Ever Infected (Cumulative):
    Plots the cumulative fraction of the population ever infected over time, for two
    different transmission rates at L = 200.

Endemic Oscillations:
    Uses FFT analysis across multiple seeds to estimate the dominant oscillation period
    of endemic infection cycles as a function of rewiring probability.

Appendix — Hub-Driven Infections:
    Tracks long-range (hub-driven) infections over time using a branching process, and
    compares the percentage of the population infected across different rewiring
    probabilities. Also plots new and cumulative hub infections side by side.

Appendix — Secondary Infections:
    Sweeps over rewiring probabilities to measure the fraction of infections attributable
    to secondary (long-range) transmission events.

Note: Code was not always executed exactly as shown — in practice, many sections were run
in smaller, isolated steps to avoid triggering large, computationally expensive simulations
in a single pass.
"""

from matplotlib import pyplot as plt
import numpy as np
from sirs_network_model import Networks
from analysis_functions import (
    compute_thresholds,
    collect_peak_stats,
    cumulative_infection_plot,
    run_fft_over_seeds,
    sweep_secondary_fraction,
    fit_models
)

plt.rcParams.update({
    "font.size": 14,          # base font size
    "axes.titlesize": 18,     # title size
    "axes.labelsize": 16,     # axis labels
    "xtick.labelsize": 14,    # x tick labels
    "ytick.labelsize": 14,    # y tick labels
    "legend.fontsize": 14,    # legend
    "figure.titlesize": 18    # figure title
})

def main():
    # These are consistent throughout
    gamma = 0.14
    omega = 0.01
    
    
    # Visuals
    A = Networks.small_world_2d_torus_k8(L=400,p=0.00, seed=0)
    t, S, I, R, _, _, final_state = Networks.run_sirs(A, alpha=0.6/8, gamma=gamma, omega=omega, I0=5, T=1000, seed=0)
    anim = Networks.snapshot_sirs_grid(A,400, 0.6/8,0.14,0.01, t_freeze=250, seed=0, I0=5,p=0)
    Networks.plot_sirs_time_series(t, S, I, R,L=400, vline=250)

    A = Networks.small_world_2d_torus_k8(L=400, p=0.01, seed=0)



    t, S, I, R, _, _, final_state = Networks.run_sirs(A, alpha=0.6/8, gamma=gamma, omega=omega, I0=5, T=1000, seed=0)
    anim = Networks.snapshot_sirs_grid(A,400, 0.6/8,0.14,0.01, t_freeze=80, seed=0, I0=5,p=0.01)
    Networks.plot_sirs_time_series(t, S, I, R,L=400, vline=80)

    A = Networks.small_world_2d_torus_k8(L=400, p=0.05, seed=0)

    t, S, I, R, _, _, final_state = Networks.run_sirs(A, alpha=0.6/8, gamma=0.14, omega=0.01, I0=5, T=1000, seed=0)
    anim = Networks.snapshot_sirs_grid(A,400, 0.6/8,0.14,0.01, t_freeze=50, seed=0, I0=5,p=0.05)
    Networks.plot_sirs_time_series(t, S, I, R,L=400, vline=50)
    

    # EPIDEMIC THRESHOLD
    # -------------------
    # Below is a theoretical way to do the entire datasets in one go. However this will take a
    # very long time to run so in practice ths was done by splitting the data into mini sets and
    # doing bit by bit. This still took over 10 hours to run on my device
    # FULL DATASET
    # One point at every 0.05
    ps = np.linspace(0.0, 1.0, 21)
    alphas = np.linspace(0.02, 0.05, 50)
    n_networks = 20
    n_seeds = 50

    all_alpha_cs = np.full((n_networks, len(ps)), np.nan)
    all_R0_cs = np.full((n_networks, len(ps)), np.nan)

    for i in range(n_networks):
        print(f"\nNetwork realisation {i+1}/{n_networks}")

        # set the network seed inside your function however you want
        alpha_c_vals, R0_c_vals = compute_thresholds(ps, alphas, n_seeds)

        all_alpha_cs[i, :] = alpha_c_vals
        all_R0_cs[i, :] = R0_c_vals

    # Mean across network realisations
    alpha_c_mean = np.nanmean(all_alpha_cs, axis=0)
    R0_c_mean = np.nanmean(all_R0_cs, axis=0)

    # Standard error across network realisations
    alpha_c_err = np.nanstd(all_alpha_cs, axis=0, ddof=1) / np.sqrt(n_networks)
    R0_c_err = np.nanstd(all_R0_cs, axis=0, ddof=1) / np.sqrt(n_networks)
    
    
    fig, ax1 = plt.subplots()

    ax1.errorbar(
        ps,
        alpha_c_mean,
        yerr=alpha_c_err,
        fmt='o',
        markersize=4,
        capsize=3
    )

    ax1.set_title('Epidemic Threshold as a Function of Rewiring Probability')
    ax1.set_xlabel('Rewiring Probability $p$')
    ax1.set_ylabel(r'Threshold $\alpha$')

    ax2 = ax1.twinx()
    ax2.set_ylim(np.array(ax1.get_ylim()) * 8 / gamma)
    ax2.set_ylabel(r'Threshold $R_0$')

    plt.show()
    
    R0_se = R0_c_std/np.sqrt(20)
    _, _, chi1, chi2,_ , res_exp, log_res = fit_models(ps, R0_c_mean, R0_se, is_small_dataset=False)
    
    # Exponential residuals
    plt.figure()
    plt.errorbar(
        ps,
        res_exp,
        yerr=R0_se,
        fmt='o',
        markersize=4,
        capsize=3
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Rewiring probability $p$')
    plt.ylabel('Residuals')
    plt.title(r'Residuals of Exponential Fit Across All $p$')
    plt.show()

    # Logarithmic residuals
    mask_log = ps > 0

    plt.figure()
    plt.errorbar(
        ps[mask_log],
        log_res,
        yerr=R0_se[mask_log],
        fmt='o',
        markersize=4,
        capsize=3
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel(r'Rewiring Probability $p$')
    plt.ylabel('Residuals')
    plt.title(r'Residuals of Logarithmic Fit Across All $p$')
    plt.show()
    
    
    # REDUCED DATASET
    ps = np.linspace(0.0, 0.05, 10) 
    
    R0_c_mean, R0_c_std, alpha_c_mean, alpha_c_std = compute_thresholds(ps)
    
    _, _, chi1_reduced_dataset, chi2_reduced_dataset,_ , res_exp, log_res = fit_models(ps, R0_c_mean, R0_c_std/np.sqrt(20), is_small_dataset=True)
    
    

    # PEAK STATISTICS
    
    Ls = [200, 300, 400]
    ps = np.linspace(0, 0.05, 10)

    results, errors, peak_times, peak_time_errors = collect_peak_stats(Ls, ps)
    plt.figure()
    for L in Ls:
        plt.errorbar(
            ps,
            results[L],
            yerr=errors[L],
            fmt='o-',
            markersize=4,
            capsize=3,
            label=f'L={L}'
        )
    plt.xlabel(r'Rewiring Probability $p$')
    plt.ylabel('Peak Percentage of Population Infected')
    plt.title('Peak Percentage Infected as a Function of Rewiring Probability')
    plt.legend()
    plt.show()

    plt.figure()
    for L in Ls:
        plt.errorbar(
            ps,
            peak_times[L],
            yerr=peak_time_errors[L],
            fmt='o-',
            markersize=4,
            capsize=3,
            label=f'L={L}'
        )
    plt.xlabel(r'Rewiring Probability $p$')
    plt.ylabel('Time of Peak Infection')
    plt.title('Time of Peak Infection as a Function of Rewiring Probability')
    plt.legend()
    plt.show()
    
    
    # EVER INFECTED
    
    cumulative_infection_plot(L=200, alpha=0.5/8)
    
    cumulative_infection_plot(L=200, alpha=0.05)
    
    
    # ENDEMIC
    
    ps = np.linspace(0, 0.05, 10)

    mean_periods = []
    se_periods = []

    for p in ps:
        fft_vals = run_fft_over_seeds(p, n_seeds=10)
        mean_periods.append(np.mean(fft_vals))
        se_periods.append(np.std(fft_vals, ddof=1) / np.sqrt(len(fft_vals)))

    mean_periods = np.array(mean_periods)
    se_periods = np.array(se_periods)

    plt.figure()
    plt.errorbar(
        ps,
        mean_periods,
        yerr=se_periods,
        fmt='o-',
        markersize=4,
        capsize=3
    )
    plt.xlabel(r'Rewiring Probability $p$')
    plt.ylabel('Mean Oscillation Period (days)')
    plt.title('Oscillation Period as a Function of Rewiring Probability')
    plt.show()
    ps = [0.0, 0.01, 0.05]
    L = 200

    fig, axes = plt.subplots(3, 1, figsize=(8,6), sharex=True)

    for i, p in enumerate(ps):

        A = Networks.small_world_2d_torus_k8(L=L, p=p, seed=1)

        t, S, I, R, new_inf, ever_mask, final_state = Networks.run_sirs(
            A,
            alpha=0.5/8,
            gamma=0.14,
            omega=0.01,
            I0=10,
            T=1500,
            seed=200,
            is_sirs=True
        )

        N = A.shape[0]
        I_frac = I / N

        axes[i].plot(t, I_frac, color="black", lw=1.5)

        axes[i].text(
            0.95, 0.9,
            f"$p = {p}$",
            transform=axes[i].transAxes,
            ha="right",
            va="top"
        )

        axes[i].set_ylabel(r"$I_{inf}(t)$")

    axes[-1].set_xlabel("Time (days)")

    plt.tight_layout()
    plt.show()
    
    # APPENDIX MEASUREMENTS
    
    # Hubs
    ps = [0.01, 0.02, 0.05]

    L = 400
    alpha = 0.5 / 8
    gamma = 0.14
    omega = 0.01
    T = 1000
    I0 = 10
    seed = 0

    # -----------------------------
    # Plot 1: Hub-driven infections
    # -----------------------------
    plt.figure()

    for p in ps:
        A = Networks.small_world_2d_torus_k8(L=L, p=p, seed=seed)

        hubs = Networks.run_branching(
            A, L, alpha, gamma, omega,
            I0=I0, T=T, seed=seed, is_sirs=False
        )

        plt.plot(np.arange(T + 1), hubs, label=f"p = {p}")

    plt.xlabel("Time (Days)")
    plt.ylabel("New Long-Range Infections Per Timestep")
    plt.title("Hub-Driven Infections Over Time")
    plt.legend()
    plt.grid(False)

    plt.show()


    # -----------------------------
    # Plot 2: % infected over time
    # -----------------------------
    plt.figure()

    for p in ps:
        A = Networks.small_world_2d_torus_k8(L=L, p=p, seed=seed)

        t, S, I, R, _, _, _ = Networks.run_sirs(
            A,
            alpha=alpha,
            gamma=gamma,
            omega=omega,
            I0=I0,
            T=T,
            seed=seed,
            is_sirs=True
        )

        I_percent = 100 * I / (L * L)

        plt.plot(t, I_percent, label=f"p = {p}")

    plt.xlabel("Time (Days)")
    plt.ylabel("Percentage of Population Infected")
    plt.title("Percentage of Population Infected Over Time")
    plt.legend()
    plt.grid(False)

    plt.show()
    

    # CUMULATIVE SEEDS (using ps and alpha etc from above)
    
    plt.figure(figsize=(8, 5))

    for p in ps:
        A = Networks.small_world_2d_torus_k8(L=L, p=p, seed=seed)

        hubs = Networks.run_branching(
            A, L, alpha, gamma, omega,
            I0=I0, T=T, seed=seed, is_sirs=False
        )

        cumulative_hubs = np.cumsum(hubs)

        plt.plot(
            np.arange(T + 1),
            hubs,
            linestyle="--",
            label=f"New hubs, p = {p}"
        )
        plt.plot(
            np.arange(T + 1),
            cumulative_hubs,
            label=f"Cumulative hubs, p = {p}"
        )

    plt.xlabel("Time (Days)")
    plt.ylabel("Number of Infections")
    plt.title("New and Cumulative Long-Range Infections Over Time")
    plt.legend()
    plt.grid(False)
    plt.show()
    
    # INFECTIONS FROM SECONDARY ONLY
    
    ps, mean_frac, std_frac = sweep_secondary_fraction(
        L=200,
        alpha=0.6/8,
        gamma=0.14,
        omega=0.01,
        I0=5,
        T=50,
        n_p=15,
        p_max=0.05,
        n_runs=10,
        is_sirs=False
    )
    
if __name__ == "__main__":
    main()
