
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter1d



class Networks:
    
    @staticmethod
    def animate_sirs_grid(A, L, beta, gamma, omega, I0=5, T=300, seed=0, interval_ms=50):
        rng = np.random.default_rng(seed)
        N = L * L
        if A.shape != (N, N):
            raise ValueError(f"A must be shape ({N},{N}) for L={L}")

        # init state
        state = np.zeros(N, dtype=np.int8)   # 0=S, 1=I, 2=R
        init = rng.choice(N, size=min(I0, N), replace=False)
        state[init] = 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("SIRS dynamics")
        ax.set_xticks([])
        ax.set_yticks([])

        # discrete colormap: S=green, I=red, R=blue
        cmap = ListedColormap(["green", "red", "blue"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        img = ax.imshow(
            state.reshape(L, L),
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        txt = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

        def update(frame):
            nonlocal state
            if frame > 0:
                state = Networks.sirs_step(A, state, beta, gamma, omega, rng)

            img.set_data(state.reshape(L, L))

            S = np.sum(state == 0)
            I = np.sum(state == 1)
            R = np.sum(state == 2)
            txt.set_text(f"t={frame}   S={S}   I={I}   R={R}")
            return img, txt

        anim = FuncAnimation(fig, update, frames=T + 1, interval=interval_ms, blit=False)
        plt.tight_layout()
        plt.show()
        return anim

    @staticmethod
    def snapshot_sirs_grid(A, L, beta, gamma, omega, t_freeze=50,
                        I0=10, seed=0, p=0):
        rng = np.random.default_rng(seed)
        N = L * L
        if A.shape != (N, N):
            raise ValueError(f"A must be shape ({N},{N}) for L={L}")

        # init state
        state = np.zeros(N, dtype=np.int8)   # 0=S, 1=I, 2=R
        init = rng.choice(N, size=min(I0, N), replace=False)
        state[init] = 1

        # evolve to t_freeze
        for t in range(1, t_freeze + 1):
            state = Networks.sirs_step(A, state, beta, gamma, omega, rng)

        # plot frozen state
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"SIRS Spread at t={t_freeze}")
        ax.set_xticks([])
        ax.set_yticks([])

        cmap = ListedColormap(["green", "red", "blue"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        ax.imshow(
            state.reshape(L, L),
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        S = np.sum(state == 0)
        I = np.sum(state == 1)
        R = np.sum(state == 2)

        ax.text(
            0.02, 0.98,
            f"t={t_freeze} days   S={100*S/N: .1f}%   I={100*I/N: .1f}%   R={100*R/N: .1f}%  p={p}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

        plt.tight_layout()
        plt.show()

        return state


    @staticmethod
    def sirs_step(A, state, beta, gamma, omega, rng, is_sirs=True):
        
        infected = (state == 1)
        susceptible = (state == 0)
        recovered = (state == 2)
        
        # infected neighbour counts for each person. 
        n_inf = np.asarray(A @ infected.astype(np.int8)).ravel()
        
        # infection probability done on paper..
        p_inf = 1.0 - (1.0 - beta) ** n_inf

        # setting wheter each person is S I or R using random probability.
        new_I = susceptible & (rng.random(len(state)) < p_inf)
        new_R = infected & (rng.random(len(state)) < gamma)
        if is_sirs:
            new_S = recovered & (rng.random(len(state)) < omega)
        else:
            new_S = 0

        # setting the new state to be fed back into the function.
        nxt = state.copy()
        nxt[new_I] = 1
        nxt[new_R] = 2
        nxt[new_S] = 0
        return nxt
        
    @staticmethod
    def run_sirs(A, beta, gamma, omega, I0=3, T=300, seed=0, init_state=None, is_sirs=True):
        """
        Runs SIRS on a fixed adjacency matrix A for T steps.

        Returns:
            t: (T+1,) array
            S, I, R: (T+1,) counts
            new_inf: (T+1,) new infections per step (S->I transitions), new_inf[0]=0
            ever_infected_mask: (N,) bool, True if node ever infected (including initial)
            state: final state array (N,)
        """
        rng = np.random.default_rng(seed)
        N = A.shape[0]

        if init_state is not None:
            state = np.array(init_state, dtype=np.int8, copy=True)
            if state.shape != (N,):
                raise ValueError(f"init_state must have shape ({N},), got {state.shape}")
        else:
            state = np.zeros(N, dtype=np.int8)
            init = rng.choice(N, size=min(I0, N), replace=False)
            state[init] = 1

        t = np.arange(T + 1)
        S = np.zeros(T + 1, dtype=int)
        I = np.zeros(T + 1, dtype=int)
        R = np.zeros(T + 1, dtype=int)

        new_inf = np.zeros(T + 1, dtype=int)  # new infections per step
        ever_infected_mask = (state == 1).copy()  # initial infected count as "ever infected"

        S[0] = np.sum(state == 0)
        I[0] = np.sum(state == 1)
        R[0] = np.sum(state == 2)

        for step in range(1, T + 1):
            prev_state = state  # keep reference to compare transitions
            state = Networks.sirs_step(A, prev_state, beta, gamma, omega, rng, is_sirs=is_sirs)

            # New infections this step: were susceptible (0), became infected (1)
            newly = (prev_state == 0) & (state == 1)
            new_inf[step] = int(np.sum(newly))

            # Update ever infected (anyone infected at any time)
            ever_infected_mask |= (state == 1)

            S[step] = np.sum(state == 0)
            I[step] = np.sum(state == 1)
            R[step] = np.sum(state == 2)

        # At the end you can compute:
        # ever_infected_count = int(ever_infected_mask.sum())
        # ever_infected_pct = 100.0 * ever_infected_count / N

        return t, S, I, R, new_inf, ever_infected_mask, state


    @staticmethod
    def plot_sirs_time_series(t, S, I, R, L, ax=None, vline=None, title="SIRS on network: I(t)"):
        """
        Standard static plot for the output of run_sirs.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        percentage_I = 100*I/L**2
        percentage_S = 100*S/L**2
        percentage_R = 100*R/L**2
        # ax.plot(t, percentage_S, label="S", linewidth=2)
        ax.plot(t, percentage_I, linewidth=2)
        # ax.plot(t, percentage_R, label="R", linewidth=2)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Percentage of Population Infected")
        ax.set_title(title)
        if vline:
            ax.axvline(x=vline, label=f't={vline} days', color='red', ls='dotted')
        ax.legend()
        plt.tight_layout()
        plt.show()

        return ax
    
    @staticmethod
    def max_fft(I, dt=1.0, burn=200, max_period=1000):
        I = np.asarray(I, float)[burn:]      # remove transient
        I = I - I.mean()
        I = I * np.hanning(len(I))

        F = np.fft.rfft(I)
        freqs = np.fft.rfftfreq(len(I), d=dt)
        amp = np.abs(F)
        amp[0] = 0.0                         # remove DC

        # frequency -> period
        mask = freqs > 0
        periods = 1.0 / freqs[mask]
        amp = amp[mask]

        # limit to max period
        mask2 = periods < max_period
        periods = periods[mask2]
        amp = amp[mask2]

        idx = np.argmax(amp)

        max_amp = amp[idx]
        dominant_period = periods[idx]

        return max_amp, dominant_period
    
    @staticmethod
    def plot_fft(I, dt=1.0, burn=200, max_period=1000, smooth_sigma=6):
        I = np.asarray(I, float)[burn:]
        I = I - I.mean()
        I = I * np.hanning(len(I))

        F = np.fft.rfft(I)
        freqs = np.fft.rfftfreq(len(I), d=dt)
        amp = np.abs(F)
        amp[0] = 0.0

        mask = freqs > 0
        freqs = freqs[mask]
        amp = amp[mask]

        # smooth in frequency space
        amp_smooth = gaussian_filter1d(amp, sigma=smooth_sigma)

        periods = 1.0 / freqs
        mask2 = periods < max_period

        plt.figure(figsize=(7,4))
        plt.plot(periods[mask2], amp_smooth[mask2])
        plt.xlabel("period (time steps)")
        plt.ylabel("amplitude")
        plt.title("Oscillation spectrum (smoothed)")
        plt.tight_layout()
        plt.show()

    
    
    @staticmethod
    def small_world_2d_torus_k8(L, p=0.1, seed=None):
        """
        2D torus Moore neighbourhood (k=8) + independent WS-style rewiring.
        Degree is NOT preserved: nodes can end up with >8 or <8.
        Each undirected edge is considered once; with prob p, rewire ONE endpoint
        to a uniformly random node (avoiding self-loops and duplicates).
        """
        rng = np.random.default_rng(seed)
        N = L * L

        def idx(x, y):
            return x * L + y

        # Build initial torus Moore-neighbour graph using sets
        adj = [set() for _ in range(N)]
        for x in range(L):
            for y in range(L):
                i = idx(x, y)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        j = idx((x + dx) % L, (y + dy) % L)
                        adj[i].add(j)
                        adj[j].add(i)

        # Consider each undirected edge once
        edges = [(i, j) for i in range(N) for j in adj[i] if i < j]

        for i, j in edges:
            # Edge might already have been rewired earlier; skip if gone
            if j not in adj[i]:
                continue
            if rng.random() >= p:
                continue

            # Rewire the edge (i, j) by keeping i fixed and changing j -> k
            # (You could instead choose randomly which endpoint to keep; see note below.)
            tries = 0
            while True:
                k = int(rng.integers(N))
                tries += 1
                if k == i:
                    continue
                if k in adj[i]:
                    continue
                break
                # (For very dense graphs you'd want a tries limit; here k=8 so it's fine.)

            # Remove old edge
            adj[i].remove(j)
            adj[j].remove(i)

            # Add new edge
            adj[i].add(k)
            adj[k].add(i)

        # Build sparse adjacency
        rows, cols = [], []
        for i in range(N):
            for j in adj[i]:
                if i != j:
                    rows.append(i)
                    cols.append(j)

        data = np.ones(len(rows), dtype=np.uint8)
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        return A

    
    


    @staticmethod
    def split_local_long(A, L):
        A = A.tocoo()
        N = L * L
        r, c = A.row, A.col

        rx, ry = divmod(r, L)
        cx, cy = divmod(c, L)

        dx = np.abs(rx - cx); dx = np.minimum(dx, L - dx)
        dy = np.abs(ry - cy); dy = np.minimum(dy, L - dy)

        is_local = (dx <= 1) & (dy <= 1) & ~((dx == 0) & (dy == 0))

        A_local = sp.coo_matrix((A.data[is_local], (r[is_local], c[is_local])), shape=(N, N)).tocsr()
        A_long  = sp.coo_matrix((A.data[~is_local], (r[~is_local], c[~is_local])), shape=(N, N)).tocsr()
        A_local.setdiag(0); A_local.eliminate_zeros()
        A_long.setdiag(0);  A_long.eliminate_zeros()
        return A_local, A_long

    @staticmethod
    def run_branching(A, L, beta, gamma, omega, I0=10, T=300, seed=0, is_sirs=False):
        rng = np.random.default_rng(seed)
        N = A.shape[0]

        A_local, A_long = Networks.split_local_long(A, L)

        state = np.zeros(N, dtype=np.int8)
        init = rng.choice(N, size=min(I0, N), replace=False)
        state[init] = 1

        hubs = np.zeros(T + 1, dtype=int)

        for t in range(1, T + 1):
            prev_state = state
            state = Networks.sirs_step(
                A, prev_state, beta, gamma, omega, rng, is_sirs=is_sirs
            )

            new_I = (prev_state == 0) & (state == 1)
            infected_prev = (prev_state == 1).astype(np.int8)

            has_long  = (np.asarray(A_long  @ infected_prev).ravel() > 0)
            has_local = (np.asarray(A_local @ infected_prev).ravel() > 0)

            hubs[t] = int(np.sum(new_I & has_long & (~has_local)))

        return hubs


    @staticmethod
    def delete_edges_random(A, q=0.1, seed=None):
        """
        Randomly delete a fraction q of UNDIRECTED edges from adjacency matrix A.
        Assumes A is symmetric 0/1. Returns a new symmetric A.
        """
        rng = np.random.default_rng(seed)
        A = A.tocsr()
        N = A.shape[0]

        # work in COO to enumerate edges once (i<j)
        C = A.tocoo()
        mask_upper = C.row < C.col
        rows = C.row[mask_upper]
        cols = C.col[mask_upper]
        m = len(rows)

        keep = rng.random(m) >= q  # keep with prob 1-q
        rows_k = rows[keep]
        cols_k = cols[keep]

        # rebuild symmetric adjacency
        r2 = np.concatenate([rows_k, cols_k])
        c2 = np.concatenate([cols_k, rows_k])
        data = np.ones(len(r2), dtype=np.uint8)

        A2 = sp.coo_matrix((data, (r2, c2)), shape=(N, N)).tocsr()
        A2.setdiag(0)
        A2.eliminate_zeros()
        return A2

    @staticmethod
    def _torus_chebyshev_dist_to_seeds(L, seed_nodes):
        """
        For each node on an LxL torus, compute Chebyshev (Moore) distance
        to the nearest node in seed_nodes, with periodic boundary conditions.
        Returns dist array of shape (N,).
        """
        N = L * L
        xs = np.arange(N) // L
        ys = np.arange(N) % L

        seed_nodes = np.asarray(list(seed_nodes), dtype=int)
        sx = seed_nodes // L
        sy = seed_nodes % L

        # Compute min Chebyshev torus distance to any seed
        dist_min = np.full(N, np.inf, dtype=float)
        for k in range(len(seed_nodes)):
            dx = np.abs(xs - sx[k])
            dx = np.minimum(dx, L - dx)
            dy = np.abs(ys - sy[k])
            dy = np.minimum(dy, L - dy)
            d = np.maximum(dx, dy)  # Chebyshev distance
            dist_min = np.minimum(dist_min, d)

        return dist_min

    @staticmethod
    def wavefront_radius_from_state(state, dist_to_seed, q=0.95):
        """
        Given a state (N,) and precomputed dist_to_seed (N,), return a scalar
        front radius based on the q-quantile of distances of infected nodes.
        """
        infected = (state == 1)
        if not np.any(infected):
            return np.nan
        return float(np.quantile(dist_to_seed[infected], q))

    @staticmethod
    def measure_wave_speed(
        A, L, beta, gamma, omega,
        I0=5, T=300, seed=0,
        is_sirs=False,
        q=0.95,
        fit_tmin=5,
        fit_tmax=60,
        min_I=5
    ):
        """
        Estimate wave speed (cells per timestep) by tracking wavefront radius
        vs time and fitting a line over [fit_tmin, fit_tmax].

        q: quantile used to define the wavefront radius (0.9-0.99 typical).
        min_I: ignore timesteps where I < min_I (front poorly defined).
        """
        rng = np.random.default_rng(seed)
        N = L * L
        if A.shape != (N, N):
            raise ValueError(f"A must be shape ({N},{N}) for L={L}")

        # init
        state = np.zeros(N, dtype=np.int8)
        init = rng.choice(N, size=min(I0, N), replace=False)
        state[init] = 1

        # distances to initial seeds (in lattice metric, not graph metric)
        dist_to_seed = Networks._torus_chebyshev_dist_to_seeds(L, init)

        t = np.arange(T + 1)
        radius = np.full(T + 1, np.nan, dtype=float)
        I_counts = np.zeros(T + 1, dtype=int)

        # t=0
        I_counts[0] = int(np.sum(state == 1))
        if I_counts[0] >= min_I:
            radius[0] = Networks.wavefront_radius_from_state(state, dist_to_seed, q=q)

        # evolve + measure
        for step in range(1, T + 1):
            state = Networks.sirs_step(A, state, beta, gamma, omega, rng, is_sirs=is_sirs)
            I_counts[step] = int(np.sum(state == 1))
            if I_counts[step] >= min_I:
                radius[step] = Networks.wavefront_radius_from_state(state, dist_to_seed, q=q)

        # fit speed on chosen window, using only finite radii
        mask = (t >= fit_tmin) & (t <= fit_tmax) & np.isfinite(radius)
        if np.sum(mask) < 2:
            return np.nan, t, radius, I_counts

        # linear fit: radius ≈ v*t + c
        v, c = np.polyfit(t[mask], radius[mask], 1)

        return float(v), t, radius, I_counts


def run_fft_over_seeds(p, n_seeds=2):
    fft_vals = []

    for seed in range(n_seeds):
        A = Networks.small_world_2d_torus_k8(L=400, p=p, seed=seed)

        t, S, I, R, _, _, _ = Networks.run_sirs(
            A,
            beta=0.5/8,
            gamma=0.14,
            omega=0.01,
            I0=10,
            T=10000,
            seed=seed # Keep constant
        )
        vals, pers =Networks.max_fft(I)
        fft_vals.append(pers)

    return np.array(fft_vals)

# p_vals = []
# mean_fft_max = []
# for i in np.linspace(0,0.5,10):
#     p = i
#     p_vals.append(p)
#     pers= run_fft_over_seeds(p=p)
#     mean_fft_max.append(pers.mean())


#     print("Mean FFT max :", pers.mean(),"p=", p)
# plt.figure()
# plt.plot(p_vals, mean_fft_max)
# plt.xlabel("Mean number of")
# plt.ylabel("Mean FFT max")
# plt.tight_layout()
# plt.show()
# A = Networks.small_world_2d_truncnorm_local(L=200, r=2, k_mean=8, k_sd=0, seed=2)
# A = Networks.small_world_2d_torus_k8(L=200, p=1, seed=1)



# # Networks.animate_sirs_grid(A, L=200, beta=0.06, gamma=0.2, omega=0.01, I0=10, T=400, seed=20)
# t, S, I, R, final_state = Networks.run_sirs(A, beta=0.4/8, gamma=0.14, omega=0.01, I0=10, T=500, seed=20)
# Networks.plot_sirs_time_series(t, S, I, R)
# Networks.plot_fft(I)