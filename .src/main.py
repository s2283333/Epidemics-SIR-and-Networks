from matplotlib import pyplot as plt
from basic_networks import Networks
from standard_SIR import TwoStrainSIRSAsym
from covid_data import CovidData


def main():

    # Covid Data
    CovidData(sheet_name='1a').plot()
    CovidData(sheet_name='1f').plot()
    CovidData(sheet_name='1j').plot()
    # SIRS
    # The figures below were done purely to get a nice "shape"
    model = TwoStrainSIRSAsym(
        N=10_000,
        I10=10,
        I20=10,
        beta1=0.2,
        beta2=0.2,  # strain 2 stronger, gives shape from paper.
        gamma1=1 / 10,
        gamma2=1 / 10,
        omega1=1 / 120,  # immunity after strain 1 lasts 120 days
        omega2=1 / 200,  # immunity after strain 2 lasts longer
        t_seed=30.0,
        t_max=1000,
        dt=0.1,
        is_R1_to_I2=True,
    )

    model.plot()

    # Networks
    N = 100
    p = 0.1

    A1 = Networks.erdos_reyni_network(N, p, seed=1)

    plt.figure(figsize=(4, 4))
    plt.spy(A1, markersize=0.1)
    plt.title("Erdős–Rényi adjacency matrix")
    plt.tight_layout()
    plt.show()

    # Using this for animation at the moment, works better.
    N = 10_000
    k = 4
    p = 0.05

    A = Networks.small_world_network(N, k, p, seed=1)
    # Visualise adjacency matrix
    plt.figure(figsize=(4, 4))
    plt.spy(A, markersize=0.1)
    plt.title("Small-world adjacency matrix")
    plt.xlabel("node j")
    plt.ylabel("node i")
    plt.tight_layout()
    plt.show()
    
    t, S, I, R, final_state = Networks.run_sirs(A, beta=0.2, gamma=0.1, omega=0.01, I0=2, T=1000, seed=2)
    Networks.plot_sirs_time_series(t, S, I, R)
    Networks.plot_fft(I)


if __name__ == "__main__":
    main()
