
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


class Networks:
    @staticmethod
    def _trunc_normal_int(rng, mean, sd, low, high):
        x = int(round(rng.normal(mean, sd)))
        return max(low, min(high, x))


    @staticmethod
    def small_world_2d_truncnorm_local(L, r, k_mean, k_sd, seed=None):
        """
        2D grid (no wrap-around) where each node is assigned a target degree k_i drawn
        from a truncated normal distribution. Undirected edges are added locally while
        respecting both endpoints' remaining degree "capacity" (no node exceeds k_i).
        Uses sets to avoid duplicates; converts to symmetric CSR adjacency at the end.

        Parameters
        ----------
        L : int
            Grid side length, N = L*L.
        r : int
            Neighbourhood radius (Chebyshev):
            r=1 -> up to 8 local neighbours
            r=2 -> up to 24 local neighbours
        k_mean, k_sd : float
            Mean and std for target degree per node (truncated to [0, max_k]).
        seed : int or None

        Returns
        -------
        A : csr_matrix (N,N)
            Symmetric adjacency matrix, zero diagonal.
        """
        rng = np.random.default_rng(seed)
        N = L * L

        def idx(x, y):
            return x * L + y

        offsets = [(dx, dy)
                for dx in range(-r, r + 1)
                for dy in range(-r, r + 1)
                if not (dx == 0 and dy == 0)]
        max_k = len(offsets)

        k_target = np.empty(N, dtype=np.int32)
        for x in range(L):
            for y in range(L):
                i = idx(x, y)
                k_target[i] = Networks._trunc_normal_int(rng, k_mean, k_sd, 0, max_k)

        adj = [set() for _ in range(N)]
        order = rng.permutation(N)

        for i in order:
            if k_target[i] <= 0:
                continue

            x, y = divmod(i, L)

            candidates = []
            for dx, dy in offsets:
                x2, y2 = x + dx, y + dy
                if 0 <= x2 < L and 0 <= y2 < L:
                    j = idx(x2, y2)
                    if j == i:
                        continue
                    if j in adj[i]:
                        continue
                    if len(adj[j]) >= k_target[j]:
                        continue
                    candidates.append(j)

            if not candidates:
                continue

            rng.shuffle(candidates)

            for j in candidates:
                if len(adj[i]) >= k_target[i]:
                    break
                if len(adj[j]) >= k_target[j]:
                    continue
                if j in adj[i]:
                    continue
                adj[i].add(j)
                adj[j].add(i)

        rows, cols = [], []
        for i in range(N):
            for j in adj[i]:
                rows.append(i)
                cols.append(j)

        data = np.ones(len(rows), dtype=np.int8)
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        return A
    
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
        ax.set_title("Infected individuals only")
        ax.set_xticks([])
        ax.set_yticks([])

        # binary image: infected = 1, others = 0
        img = ax.imshow(
            (state == 1).reshape(L, L),
            vmin=0, vmax=1,
            cmap="Reds",
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

            infected_grid = (state == 1).reshape(L, L)
            img.set_data(infected_grid)

            I = np.sum(state == 1)
            txt.set_text(f"t={frame}   I={I}")
            return img, txt

        anim = FuncAnimation(fig, update, frames=T + 1, interval=interval_ms, blit=False)
        plt.tight_layout()
        plt.show()
        return anim

    @staticmethod
    def small_world_network(N, k, p, seed=None, varying_neighbours=True):
        """
        N : number of nodes
        k : each node connected to k nearest neighbours (k even)
        p : rewiring probability
        varying_neighbours : trialling the theory that not everyone has the same number of contacts(random)
        """
        rng = np.random.default_rng(seed)

        # Data to be stored in rows and columns eg if i=10 and i=11 are connected
        # then row=10 column=11.
        rows = []
        cols = []

        for i in range(N):
            
            # Setting differing numbers of neighbours(if varying_neighbours is true)
            if varying_neighbours:
                k_i = max(0, int(round(k * rng.normal(loc=1.0, scale=2))))  # scale controls variation
            else:
                k_i = k
                
            half = k_i // 2
            
            # giving connections (only forwards).
            for d in range(1, half + 1):
                
                j = (i + d) % N
                # Edge case resolving
                if i < j:
                    rows.append(i)
                    cols.append(j)
                else:
                    rows.append(j)
                    cols.append(i)

        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)

        
        # Looping over all edges 
        for idx in range(len(rows)):
            
            # randomly changing some to be long range
            if rng.random() < p:
                i = rows[idx]
                j = cols[idx]

                # pick a new endpoint m for edge (i, j)
                # avoid self-loops and duplicate edges
                while True:
                    m = rng.integers(0, N)
                    # Checks we dont create any double edges (or have a self connecting network)
                    a, b = min(i, m), max(i, m)
                    if m!=i and not ((rows == a) & (cols == b)).any():
                        break

                # rewire edge to (i, m)
                rows[idx] = min(i, m)
                cols[idx] = max(i, m)
                
        data = np.ones(len(rows), dtype=np.int8)

        # matrix formation. Ensuring symetry
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        A = A + A.T

        return A
    
    
    @staticmethod
    def erdos_reyni_network(N, p, seed=None):
        """
        N : number of nodes
        p : probability of an edge between any pair
        """
        rng = np.random.default_rng(seed)

        rows = []
        cols = []

        # Only loop over i < j to avoid duplicates
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < p:
                    rows.append(i)
                    cols.append(j)

        data = np.ones(len(rows), dtype=np.int8)

        # Make symmetric adjacency matrix
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        A = A + A.T
        return A

    @staticmethod
    def sirs_step(A, state, beta, gamma, omega, rng):
        
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
        new_S = recovered & (rng.random(len(state)) < omega)

        # setting the new state to be fed back into the function.
        nxt = state.copy()
        nxt[new_I] = 1
        nxt[new_R] = 2
        nxt[new_S] = 0
        return nxt
    
    @staticmethod
    def run_sirs(A, beta, gamma, omega, I0=3, T=300, seed=0, init_state=None):
        """
        Runs SIRS on a fixed adjacency matrix A for T steps.

        Returns:
            t: (T+1,) array
            S: (T+1,) counts
            I: (T+1,) counts
            R: (T+1,) counts
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

        S[0] = np.sum(state == 0)
        I[0] = np.sum(state == 1)
        R[0] = np.sum(state == 2)

        for step in range(1, T + 1):
            state = Networks.sirs_step(A, state, beta, gamma, omega, rng)
            S[step] = np.sum(state == 0)
            I[step] = np.sum(state == 1)
            R[step] = np.sum(state == 2)

        return t, S, I, R, state

    @staticmethod
    def plot_sirs_time_series(t, S, I, R, ax=None, title="SIRS on network: S(t), I(t), R(t)"):
        """
        Standard static plot for the output of run_sirs.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        # ax.plot(t, S, label="S", linewidth=2)
        ax.plot(t, I, label="I", linewidth=2)
        # ax.plot(t, R, label="R", linewidth=2)
        ax.set_xlabel("time step")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

        return ax

    @staticmethod
    # ---Below is (mostly) AI generated just so i could get a visual understanding of what was happening---
    def animate_sirs(A, beta, gamma, omega, I0=2, T=200, seed=0, interval_ms=100):
        rng = np.random.default_rng(seed)
        N = A.shape[0]

        # circle layout for clarity
        ang = np.linspace(0, 2*np.pi, N, endpoint=False)
        xy = np.c_[np.cos(ang), np.sin(ang)]

        # draw edges once (upper triangle only)
        A_up = sp.triu(A, k=1).tocoo()
        ex, ey = [], []
        for i, j in zip(A_up.row, A_up.col):
            ex += [xy[i, 0], xy[j, 0], np.nan]
            ey += [xy[i, 1], xy[j, 1], np.nan]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot(ex, ey, linewidth=0.6, alpha=0.35)

        # init state
        state = np.zeros(N, dtype=np.int8)
        init = rng.choice(N, size=I0, replace=False)
        state[init] = 1

        scat = ax.scatter(xy[:, 0], xy[:, 1], s=90)
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

        def state_colors(st):
            c = np.empty(N, dtype=object)
            c[st == 0] = "tab:blue"   # S
            c[st == 1] = "tab:red"    # I
            c[st == 2] = "tab:green"  # R
            return c

        def update(frame):
            nonlocal state
            state = Networks.sirs_step(A, state, beta, gamma, omega, rng)

            scat.set_color(state_colors(state))
            S = np.sum(state == 0)
            I = np.sum(state == 1)
            R = np.sum(state == 2)
            txt.set_text(f"t={frame:3d}   S={S}  I={I}  R={R}")
            return scat, txt

        anim = FuncAnimation(fig, update, frames=T+1, interval=interval_ms, blit=False)
        plt.show()
        return anim

    @staticmethod
    def animate_I_time_series(A, beta, gamma, omega, I0=3, T=300, seed=0, interval_ms=1, later_focus_time=None):
        rng = np.random.default_rng(seed)
        N = A.shape[0]

        state = np.zeros(N, dtype=np.int8)
        init = rng.choice(N, size=min(I0, N), replace=False)
        state[init] = 1

        t_vals = np.arange(T + 1)
        I_vals = np.zeros(T + 1, dtype=int)
        I_vals[0] = np.sum(state == 1)

        fig, ax = plt.subplots(figsize=(7, 4))

        if later_focus_time is not None:
            ax.set_xlim(later_focus_time, T)
            ax.set_ylim(0, N // 10)
        else:
            ax.set_xlim(0, T)
            ax.set_ylim(0, N)

        ax.set_xlabel("time step")
        ax.set_ylabel("infected")
        ax.set_title("SIRS on network: I(t)")

        (line,) = ax.plot([], [], linewidth=2)
        txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

        def update(frame):
            nonlocal state

            if frame > 0:
                state = Networks.sirs_step(A, state, beta, gamma, omega, rng)
                I_vals[frame] = np.sum(state == 1)

            if later_focus_time is not None:
                mask = t_vals[:frame+1] >= later_focus_time
                x = t_vals[:frame+1][mask]
                y = I_vals[:frame+1][mask]
            else:
                x = t_vals[:frame+1]
                y = I_vals[:frame+1]

            line.set_data(x, y)
            txt.set_text(f"t={frame}   I={I_vals[frame]}")
            return line, txt

        anim = FuncAnimation(
            fig, update,
            frames=T + 1,
            interval=interval_ms,
            blit=False
        )

        plt.tight_layout()
        plt.show()
        return anim
    
    @staticmethod
    def max_fft(I, dt=1.0, burn=200, max_period=1000):
        I = np.asarray(I, float)[burn:]   # remove transient
        I = I - I.mean()
        I = I * np.hanning(len(I))

        F = np.fft.rfft(I)
        freqs = np.fft.rfftfreq(len(I), d=dt)
        amp = np.abs(F)
        amp[0] = 0.0                      # remove DC

        # convert frequency -> period
        mask = freqs > 0
        periods = 1.0 / freqs[mask]
        mask2 = periods < 400
        amp1 = amp[mask]
        amp= amp1[mask2]

        return max(np.abs(amp))
    
    @staticmethod
    def plot_fft(I, dt=1.0, burn=200, max_period=1000):
        I = np.asarray(I, float)[burn:]   # remove transient
        I = I - I.mean()
        I = I * np.hanning(len(I))

        F = np.fft.rfft(I)
        freqs = np.fft.rfftfreq(len(I), d=dt)
        amp = np.abs(F)
        amp[0] = 0.0                      # remove DC

        # convert frequency -> period
        mask = freqs > 0
        periods = 1.0 / freqs[mask]
        amp = amp[mask]
        mask2 = periods < 400
        amp2 = amp[mask2]
        print(amp2.max())
        plt.figure(figsize=(7,4))
        plt.plot(periods, amp)
        plt.xlim(0, max_period)
        plt.xlabel("period (time steps)")
        plt.ylabel("amplitude")
        plt.title("Oscillation spectrum (period domain)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def small_world_2d_torus_k8(L, p=0.1, seed=None):
        """Sets up 2d small world network 

        Args:
            L (int): size of each side of the grid
            p (float, optional): probability of a connection getting rewired
            seed (int, optional): seed, to make deterministic, good to be able to vary

        Returns:
            sp.coo_matrix: connections matrix.
        """
        rng = np.random.default_rng(seed)
        N = L * L

        def idx(x, y):
            """gives a unique id to each node based on position in the grid"""
            return x * L + y

        # sets of connections between nodes
        adj = [set() for _ in range(N)]

        # Moore neighbourhood on a torus (degree = 8)
        for x in range(L):
            for y in range(L):
                i = idx(x, y)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx or dy:
                            # Sorting edge effects
                            j = idx((x + dx) % L, (y + dy) % L)
                            adj[i].add(j)
                            adj[j].add(i)

        # undirected edges
        edges = [(i, j) for i in range(N) for j in adj[i] if i < j]

        # degree-preserving rewiring (milder: mostly local swaps, occasional long-range)
        for i, j in edges:
            if j not in adj[i] or rng.random() >= p:
                continue


            k = int(rng.integers(N))
            if k == i or k == j:
                continue

            if k in adj[i]:
                continue

            # pick l among neighbours of k (always local on torus)
            l = int(rng.choice(list(adj[k])))
            if l == i or j in adj[l] or l == j:
                continue

            adj[i].remove(j); adj[j].remove(i)
            adj[k].remove(l); adj[l].remove(k)
            adj[i].add(k); adj[k].add(i)
            adj[j].add(l); adj[l].add(j)

        rows = [i for i in range(N) for j in adj[i] if i>j]
        cols = [j for i in range(N) for j in adj[i] if i>j]


        A = sp.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(N, N))
        return A + A.T
# 1. Make into proper class structure rather than just name space. with run through storing the relevant data (not just animation)
# 2. Look at impact of varying parameters. Number of neighbours. Randomness variation. Disease parameters.
# 3. Apply data to optimise parameters.
# 4. Incorporate multiple waves of disease potentially?
# 5. If looking at post restrictions era try modelling with an artificial starting point at that date.

# Things of interest:
# 1. Higher variation in number of connections leads to more infections
# 2. It also leads to less notable oscillations
 
def run_fft_over_seeds(p, n_seeds=10):
    fft_vals = []

    for seed in range(n_seeds):
        A = Networks.small_world_2d_torus_k8(L=200, p=p, seed=seed)

        t, S, I, R, _ = Networks.run_sirs(
            A,
            beta=0.06,
            gamma=0.2,
            omega=0.01,
            I0=10,
            T=1000,
            seed=seed # Keep constant
        )

        fft_vals.append(Networks.max_fft(I))

    return np.array(fft_vals)

# p_vals = []
# mean_fft_max = []
# for i in range(0,9):
#     p = i/10
#     p_vals.append(p)
#     vals = run_fft_over_seeds(p=p)
#     mean_fft_max.append(vals.mean())


#     print("Mean FFT max :", vals.mean(),"p=", p)
# plt.figure()
# plt.plot(p_vals, mean_fft_max)
# plt.xlabel("Mean number of")
# plt.ylabel("Mean FFT max")
# plt.tight_layout()
# plt.show()
# A = Networks.small_world_2d_truncnorm_local(L=200, r=2, k_mean=8, k_sd=0, seed=2)
A = Networks.small_world_2d_torus_k8(L=200, p=1, seed=1)



# Networks.animate_sirs_grid(A, L=200, beta=0.06, gamma=0.2, omega=0.01, I0=10, T=400, seed=20)
t, S, I, R, final_state = Networks.run_sirs(A, beta=0.4/8, gamma=0.14, omega=0.01, I0=10, T=500, seed=20)
Networks.plot_sirs_time_series(t, S, I, R)
Networks.plot_fft(I)