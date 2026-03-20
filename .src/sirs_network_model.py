"""
A module for simulating epidemic spread (SIRS/SIR models) on small-world 2D torus networks.
This module provides tools to:
- Generate small-world networks with tunable rewiring probability
- Simulate SIRS/SIR dynamics on arbitrary network topologies
- Visualise and analyse epidemic spread patterns
- Track infection lineages (primary vs. secondary transmission)
- Perform spectral analysis of infection time series
Class:
    Networks: Namespace containing network generation, epidemic simulation, visualisation, and analysis.
Key Methods:
    - small_world_2d_torus_k8(): Generate a 2D small-world network with Moore neighbourhood + rewiring
    - sirs_step(): Perform one time step of SIRS/SIR dynamics
    - run_sirs(): Simulate complete epidemic trajectory
    - animate_sirs_grid(): Real-time animation of spatiotemporal epidemic spread
    - snapshot_sirs_grid(): Capture epidemic state at a specific time point
    - plot_sirs_time_series(): Plot infection prevalence over time
    - max_fft(): Compute dominant oscillation period via FFT
    - split_local_long(): Decompose network into local and long-range edges
    - run_branching(): Track infections caused purely by long-range transmission
    - run_sirs_lineage(): Track primary vs. secondary infection lineages
"""

from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm



class Networks:
    
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
            """Returns unique ID for each node"""
            return x * L + y

        # Build initial torus Moore-neighbour graph using sets
        adj = [set() for _ in range(N)]
        
        
        for x in range(L):
            for y in range(L):
                i = idx(x, y)
                
                # Here we add initial connections
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        
                        # Preventing wiring to itself
                        if dx == 0 and dy == 0:
                            continue
                        
                        # Adding conneccted node as j
                        j = idx((x + dx) % L, (y + dy) % L)
                        
                        # For symmetry purposes we add both.
                        adj[i].add(j)
                        adj[j].add(i)

        # Consider each undirected edge once
        edges = [(i, j) for i in range(N) for j in adj[i] if i < j]

        for i, j in edges:
            # Edge might already have been rewired earlier, skip if gone
            if j not in adj[i]:
                continue
            # Check random number between 0 and 1 against rewiring probability
            if rng.random() >= p:
                continue

            # Rewire the edge (i, j) by keeping i fixed and changing j -> k
            # This creates tiny edge effect where very very high level nodes will not have edges rewired. 
            # But in a large system this effect is negligble.
            tries = 0
            while True:
                k = int(rng.integers(N))
                
                # Edge case of rewiring to itself
                if k == i:
                    continue
                # Edge case of rewiring to an already wired node
                if k in adj[i]:
                    continue
                break

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
    def sirs_step(A, state, alpha, gamma, omega, rng, is_sirs=True):
        """
        Perform one time-step update of the SIRS (or SIR) model on a network.

        Nodes transition as:
        - S → I based on infected neighbours (alpha)
        - I → R with probability gamma
        - R → S with probability omega (if is_sirs=True)

        Args:
            A: Adjacency matrix
            state: Array of node states (0=S, 1=I, 2=R)
            alpha: Infection rate
            gamma: Recovery rate
            omega: Loss of immunity rate
            rng: Random number generator
            is_sirs: If False, disables R → S transition

        Returns:
            Updated state array
        """
        
        # Masks of each state
        infected = (state == 1)
        susceptible = (state == 0)
        recovered = (state == 2)
        
        # infected neighbour counts for each person. 
        n_inf = np.asarray(A @ infected.astype(np.int8)).ravel()
        
        # infection probability 
        p_inf = 1.0 - (1.0 - alpha) ** n_inf

        # setting wheter each person is S I or R using random probability.
        new_I = susceptible & (rng.random(len(state)) < p_inf)
        new_R = infected & (rng.random(len(state)) < gamma)
        
        # Optional trigger between SIR and SIRS
        if is_sirs:
            new_S = recovered & (rng.random(len(state)) < omega)
        else:
            new_S = 0

        # setting the new state to be fed back into the function
        nxt = state.copy()
        nxt[new_I] = 1
        nxt[new_R] = 2
        nxt[new_S] = 0
        return nxt
        
    @staticmethod
    def run_sirs(A, alpha, gamma, omega, I0=3, T=300, seed=0, init_state=None, is_sirs=True):
        """
        Runs SIRS on a fixed adjacency matrix A for T steps.
        init_state gives option for manual input of state

        Returns:
            t: (T+1,) array
            S, I, R: (T+1,) counts
            new_inf: (T+1,) new infections per step
            ever_inf_frac: (T+1,) cumulative fraction ever infected
            state: final state array (N,)
        """
        rng = np.random.default_rng(seed)
        N = A.shape[0]

        # Checking if 
        if init_state is not None:
            state = np.array(init_state, dtype=np.int8, copy=True)
            if state.shape != (N,):
                raise ValueError(f"init_state must have shape ({N},), got {state.shape}")
        else:
            # All susceptible
            state = np.zeros(N, dtype=np.int8)
            
            # I0 random nodes infected
            init = rng.choice(N, size=min(I0, N), replace=False)
            state[init] = 1

        # Setting up arrays which we want to track
        t = np.arange(T + 1)
        S = np.zeros(T + 1, dtype=int)
        I = np.zeros(T + 1, dtype=int)
        R = np.zeros(T + 1, dtype=int)

        # Number of newly infected individuals per step
        new_inf = np.zeros(T + 1, dtype=int)

        # Track cumulative infections
        ever_infected_mask = (state == 1).copy()
        ever_inf_frac = np.zeros(T + 1, dtype=float)
        ever_inf_frac[0] = np.sum(ever_infected_mask) / N

        S[0] = np.sum(state == 0)
        I[0] = np.sum(state == 1)
        R[0] = np.sum(state == 2)

        # Step through the infection
        for step in range(1, T + 1):
            prev_state = state
            state = Networks.sirs_step(A, prev_state, alpha, gamma, omega, rng, is_sirs=is_sirs)

            # New infections this step
            newly = (prev_state == 0) & (state == 1)
            new_inf[step] = int(np.sum(newly))

            # Update cumulative infected
            ever_infected_mask |= (state == 1)
            ever_inf_frac[step] = np.sum(ever_infected_mask) / N

            S[step] = np.sum(state == 0)
            I[step] = np.sum(state == 1)
            R[step] = np.sum(state == 2)

        return t, S, I, R, new_inf, ever_inf_frac, state
    
    
    @staticmethod
    def animate_sirs_grid(A, L, alpha, gamma, omega, I0=10, T=300, seed=0, interval_ms=50, is_sirs=True):
        """
        Animates SIR/SIRS dynamics on a 2D grid network in real-time.

        Args:
            A (scipy.sparse matrix): Adjacency matrix of shape (N, N) where N = L*L
            L (int): Side length of the 2D square lattice (torus topology)
            alpha (float): Transmission rate per infected neighbor
            gamma (float): Recovery rate (probability of I -> R per step)
            omega (float): Loss of immunity rate (probability of R -> S per step)
            I0 (int, optional): Number of initially infected nodes. Defaults to 10.
            T (int, optional): Total number of timesteps to simulate. Defaults to 300.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            interval_ms (int, optional): Delay between animation frames in milliseconds. Defaults to 50.
            is_sirs (bool, optional): If True, run SIRS. If False, run SIR. Defaults to True.

        Raises:
            ValueError: If A is not shape (N, N) for the given L.

        Returns:
            anim (FuncAnimation): Matplotlib animation object showing S/I/R states over time.
        """
        rng = np.random.default_rng(seed)
        N = L * L
        if A.shape != (N, N):
            raise ValueError(f"A must be shape ({N},{N}) for L={L}")

        state = np.zeros(N, dtype=np.int8)   # 0=S, 1=I, 2=R
        init = rng.choice(N, size=I0, replace=False)
        state[init] = 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("SIRS dynamics" if is_sirs else "SIR dynamics")
        ax.set_xticks([])
        ax.set_yticks([])

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
                state = Networks.sirs_step(A, state, alpha, gamma, omega, rng, is_sirs=is_sirs)

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
    def snapshot_sirs_grid(A, L, alpha, gamma, omega, t_freeze=50,
                        I0=10, seed=0, p=0):
        """Takes a snapshot of the animated grid at a chosen time (t_freeze)"""
        
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
            state = Networks.sirs_step(A, state, alpha, gamma, omega, rng)

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
    def plot_sirs_time_series(t, S, I, R, L, ax=None, vline=None, title="Percentage of Population Infected Over Time"):
        """
        Standard static plot for the output of run_sirs.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            
        percentage_I = 100*I/L**2
        
        # We don't plot S and R as they often cloud the I part of the graph.
        # percentage_S = 100*S/L**2
        # percentage_R = 100*R/L**2
        
        ax.plot(t, percentage_I, linewidth=2)
        # ax.plot(t, percentage_S, label="S", linewidth=2)
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
    def split_local_long(A, L):
        """
        Split the adjacency matrix into local (nearest-neighbour) and long-range edges.

        Local = Moore neighbourhood on the LxL torus (i.e. the original lattice).
        Long-range = everything else (i.e. rewired edges).
        """

        # Work in COO so we can access edge lists directly
        A = A.tocoo()
        N = L * L
        r, c = A.row, A.col   # row/col indices of edges

        # Convert node indices into 2D grid coordinates
        rx, ry = divmod(r, L)
        cx, cy = divmod(c, L)

        # Compute distances with periodic boundary conditions (torus)
        dx = np.abs(rx - cx)
        dx = np.minimum(dx, L - dx)

        dy = np.abs(ry - cy)
        dy = np.minimum(dy, L - dy)

        # Local edges = within 1 step in both x and y (Moore neighbourhood)
        # Exclude self-connections just in case
        is_local = (dx <= 1) & (dy <= 1) & ~((dx == 0) & (dy == 0))

        # Build separate adjacency matrices using masks
        A_local = sp.coo_matrix(
            (A.data[is_local], (r[is_local], c[is_local])),
            shape=(N, N)
        ).tocsr()

        A_long = sp.coo_matrix(
            (A.data[~is_local], (r[~is_local], c[~is_local])),
            shape=(N, N)
        ).tocsr()

        # Clean up any accidental self-loops / zeros
        A_local.setdiag(0)
        A_local.eliminate_zeros()

        A_long.setdiag(0)
        A_long.eliminate_zeros()

        return A_local, A_long

    @staticmethod
    def run_branching(A, L, alpha, gamma, omega, I0=10, T=300, seed=0, is_sirs=False):
        """
        Run the model and track infections caused purely by long-range edges.

        At each time step, counts how many new infections occur where:
        - the node had a long-range infected neighbour
        - but no local infected neighbours

        Returns an array of long range-driven infections over time.
        """

        rng = np.random.default_rng(seed)
        N = A.shape[0]

        # Split network into local (lattice) and long-range (rewired) edges
        A_local, A_long = Networks.split_local_long(A, L)

        # Initialise state with I0 random infected nodes
        state = np.zeros(N, dtype=np.int8)
        init = rng.choice(N, size=min(I0, N), replace=False)
        state[init] = 1

        # Store number of "branching" infections at each time
        hubs = np.zeros(T + 1, dtype=int)

        for t in range(1, T + 1):
            prev_state = state

            # Advance system one time step
            state = Networks.sirs_step(
                A, prev_state, alpha, gamma, omega, rng, is_sirs=is_sirs
            )

            # Newly infected nodes (S → I)
            new_I = (prev_state == 0) & (state == 1)

            # Nodes that were infected in previous step
            infected_prev = (prev_state == 1).astype(np.int8)

            # Check if each node had infected neighbours via long / local edges
            has_long  = (np.asarray(A_long  @ infected_prev).ravel() > 0)
            has_local = (np.asarray(A_local @ infected_prev).ravel() > 0)

            # Count infections caused purely by long-range connections
            hubs[t] = int(np.sum(new_I & has_long & (~has_local)))

        return hubs
    
    
    @staticmethod
    def run_sirs_lineage(A, L, alpha, gamma, omega, I0=10, T=300, seed=0, is_sirs=False):
        """
        Runs SIRS/SIR and tracks what fraction of ever-infected nodes belong to
        the secondary lineage (infections tracing purely to long-range seeds).

        Attribution is two-stage per step:
            Stage 1 — Direct secondary seeds: newly infected via a long-range
                    contact with no local infected neighbour present.
            Stage 2 — Secondary descendants: newly infected whose only infected
                    neighbours are already in the secondary lineage.
        Primary wins all ties (ambiguous nodes labelled 0).
        
        Returns the standard variables (as per run_sirs) alongside fraction from secondary
        infections.

        """
        rng = np.random.default_rng(seed)
        N = A.shape[0]

        # Split once, reuse every step
        A_local, A_long = Networks.split_local_long(A, L)

        state = np.zeros(N, dtype=np.int8)
        init  = rng.choice(N, size=min(I0, N), replace=False)
        state[init] = 1

        # -1=never infected, 0=primary (or ambiguous), 1=secondary
        lineage_labels = np.full(N, -1, dtype=np.int8)
        lineage_labels[init] = 0

        t = np.arange(T + 1)
        S = np.zeros(T + 1, dtype=int)
        I = np.zeros(T + 1, dtype=int)
        R = np.zeros(T + 1, dtype=int)
        new_inf        = np.zeros(T + 1, dtype=int)
        ever_inf_frac  = np.zeros(T + 1, dtype=float)
        frac_secondary = np.zeros(T + 1, dtype=float)

        # t=0
        sc = np.bincount(state.astype(np.uint8), minlength=3)
        S[0], I[0], R[0] = sc[0], sc[1], sc[2]
        ever_inf_frac[0] = min(I0, N) / N
        # frac_secondary[0] = 0 by default

        for step in range(1, T + 1):
            prev_state    = state.copy()
            infected_prev = (prev_state == 1)

            state = Networks.sirs_step(
                A, prev_state, alpha, gamma, omega, rng, is_sirs=is_sirs
            )

            newly = (prev_state == 0) & (state == 1)
            new_inf[step] = int(newly.sum())

            if new_inf[step] > 0:
                infected_prev_int = infected_prev.astype(np.int8)

                # Stage 1: direct secondary seeds
                # newly infected via long-range contact only (no local infected neighbour)
                has_any_long  = (np.asarray(A_long  @ infected_prev_int).ravel() > 0)
                has_any_local = (np.asarray(A_local @ infected_prev_int).ravel() > 0)
                direct_secondary = newly & has_any_long & ~has_any_local

                # Stage 2: descendants of existing secondary lineage nodes
                # secondary parent present, no primary parent
                is_primary_inf   = (infected_prev & (lineage_labels == 0)).astype(np.int8)
                is_secondary_inf = (infected_prev & (lineage_labels == 1)).astype(np.int8)
                has_primary_parent   = (np.asarray(A @ is_primary_inf).ravel()   > 0)
                has_secondary_parent = (np.asarray(A @ is_secondary_inf).ravel() > 0)
                secondary_descendant = newly & has_secondary_parent & ~has_primary_parent

                # Stage 3: connected to infected with primary lineage and infected with secondary
                # In this split we give to primary
                pure_s = direct_secondary | secondary_descendant

                lineage_labels[newly & ~pure_s] = 0
                lineage_labels[pure_s]          = 1

            # Compartment counts
            sc = np.bincount(state.astype(np.uint8), minlength=3)
            S[step], I[step], R[step] = sc[0], sc[1], sc[2]

            # Ever-infected and secondary fraction
            ever_lc = np.bincount(lineage_labels[lineage_labels >= 0], minlength=2)
            total = int(ever_lc.sum())
            ever_inf_frac[step]  = total / N
            frac_secondary[step] = ever_lc[1] / total if total > 0 else 0.0

        return t, S, I, R, new_inf, ever_inf_frac, frac_secondary, state