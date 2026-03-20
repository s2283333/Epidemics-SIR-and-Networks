"""
standard_SIR:
This file is not used however was maintained in the github.
It sets up a two strain stochastic SIR(S) model.
This was mainly done in an alternate exploratory route into how two strains compete.
Only ~10% of people ever got the initial strain of COVID. One pertinent theory for this was
that the second disease "took over" as it was more infectious.
This code enables one to explore different disease parameters across the two strains.
Although none of this is used in the report it does provide interesting results that 
both strains will infect a very high proportion of the population regardless of the presence 
of another disease. 
We also allow an movement form R1 -> I2 directly effectively implying recovery from strain 1
does not provide immunity to strain 2. 
"""
from matplotlib.pylab import False_
import numpy as np
import matplotlib.pyplot as plt


class TwoStrainSIRSAsym:
    """
    Two-strain SIRS with asymmetric cross-immunity:
      - After I1 -> R1: immune to strain 1 only; still susceptible to strain 2
      - After I2 -> R2: immune to both strains
      - R1 and R2 wane back to S at rates omega1, omega2
      - Strain 2 is seeded at time t_seed by adding I20 infections
      - Optional filter to push R1 -> I2. This better replicates the desired shape
    """

    def __init__(
        self,
        N,
        I10,
        I20,
        beta1,
        beta2,
        gamma1,
        gamma2,
        omega1,  # waning from R1 -> S
        omega2,  # waning from R2 -> S
        t_seed,  # when I2 introduced 
        t_max,
        dt,
        R10=0.0,
        R20=0.0,  
        is_R1_to_I2=False,
        stochastic_intervention=True,
        interventions=[]
    ):
        self.N = N
        self.I10 = I10
        self.I20 = I20
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.omega1 = omega1
        self.omega2 = omega2
        self.t_seed = t_seed
        self.t_max = t_max
        self.dt = dt
        self.R10 = R10
        self.R20 = R20
        self.is_R1_to_I2 = is_R1_to_I2
        self.t = None
        self.stochastic_intervention = stochastic_intervention
        self.rng = np.random.default_rng()
        self.sigma= 0.4
        self.interventions = interventions or []
    
    def run(self):
        
        t = np.arange(0, self.t_max + self.dt, self.dt)
        n = len(t)

        S = np.zeros(n, dtype=float)
        I1 = np.zeros(n, dtype=float)
        I2 = np.zeros(n, dtype=float)
        R1 = np.zeros(n, dtype=float)
        R2 = np.zeros(n, dtype=float)
        new1 = np.zeros(n, dtype=float) # new infections into I1 per step (counts)
        new2 = np.zeros(n, dtype=float) # new infections into I2 per step (counts)
        
        
        # Initial conditions (strain 2 starts at 0; we seed later)
        S[0] = self.N - self.I10 - self.R10 - self.R20
        I1[0] = new1[0] = self.I10
        I2[0] = new2[0] = 0.0
        R1[0] = self.R10
        R2[0] = self.R20

        seed_k = int(round(self.t_seed / self.dt))

        for k in range(n - 1):
            s, i1, i2, r1, r2 = S[k], I1[k], I2[k], R1[k], R2[k]
            tk = t[k]
            # Inputting second strain
            if k == seed_k:
                # First people infected by second strain come from susceptible, keeps N constant.
                s -= self.I20
                i2 += self.I20
                new2[k] += self.I20
            if self.stochastic_intervention:
                xi1 = self.rng.normal(0.0, 1.0)
                xi2 = self.rng.normal(0.0, 1.0)
                beta1 = max(0.0, self.beta1 * (1 + self.sigma * xi1))
                beta2 = max(0.0, self.beta2 * (1 + self.sigma * xi2))
                
            else:
                beta1 = self.beta1
                beta2 = self.beta2
                
            for (t0, t1, m1, m2) in self.interventions:
                if (tk >= t0) and (tk < t1):
                    beta1 *= m1
                    beta2 *= m2
            # Forces of infection (mean-field)
            lam1 = beta1 * i1 / self.N
            lam2 = beta2 * i2 / self.N

            # Flows
            S_to_I1 = lam1 * s
            S_to_I2 = lam2 * s
            if self.is_R1_to_I2:
                R1_to_I2 = lam2 * r1  # allowing people unsusceptible to strain 1 to go straight to strain 2.
            else:
                R1_to_I2 = 0
            I1_to_R1 = self.gamma1 * i1
            I2_to_R2 = self.gamma2 * i2

            R1_to_S = self.omega1 * r1
            R2_to_S = self.omega2 * r2
            # Record incidence as counts over this timestep
            new1[k] = self.dt * S_to_I1
            new2[k] = self.dt * (S_to_I2 + R1_to_I2)

            # Derivatives
            dS = -S_to_I1 - S_to_I2 + R1_to_S + R2_to_S
            dI1 = S_to_I1 - I1_to_R1
            dI2 = S_to_I2 + R1_to_I2 - I2_to_R2
            dR1 = I1_to_R1 - R1_to_I2 - R1_to_S
            dR2 = I2_to_R2 - R2_to_S
            

            # Euler step
            S[k + 1] = s + self.dt * dS
            I1[k + 1] = i1 + self.dt * dI1
            I2[k + 1] = i2 + self.dt * dI2
            R1[k + 1] = r1 + self.dt * dR1
            R2[k + 1] = r2 + self.dt * dR2

            # Optional: clamp small numerical negatives
            S[k + 1] = max(0.0, S[k + 1])
            I1[k + 1] = max(0.0, I1[k + 1])
            I2[k + 1] = max(0.0, I2[k + 1])
            R1[k + 1] = max(0.0, R1[k + 1])
            R2[k + 1] = max(0.0, R2[k + 1])

        self.t, self.S, self.I1, self.I2, self.R1, self.R2 = t, S, I1, I2, R1, R2
        self.new1, self.new2 = new1, new2
        return t, S, I1, I2, R1, R2

    def plot(self):
        if self.t is None:
            self.run()

        # plt.plot(self.t, self.S, label="S")
        plt.plot(self.t, self.I1, label="I1")
        plt.plot(self.t, self.I2, label="I2")
        # plt.plot(self.t, self.R1, label="R1 (immune to 1 only)")
        # plt.plot(self.t, self.R2, label="R2 (immune to both)")
        plt.axvline(self.t_seed, linestyle="--", linewidth=1, label="seed strain 2")
        plt.xlabel("time (days)")
        plt.ylabel("people")
        plt.title("Two-strain SIRS with asymmetric cross-immunity")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def i1_peak_infections(self):
        """
        Returns:
        k_peak: index of peak I1
        t_peak: time at peak
        I1_peak: peak value
        cum1_to_peak: cumulative infections of strain 1 up to peak (counts)
        cum2_to_peak: cumulative infections of strain 2 up to peak (counts)
        cum_total_to_peak: total cumulative infections up to peak (counts)
        frac_total_to_peak: total cumulative infections up to peak as fraction of N
        Requires: run() has been called and stores self.new1, self.new2.
        """
        k_peak = int(np.argmax(self.I1))
        cum1_to_peak = float(np.sum(self.new1[:k_peak + 1]))
        cum2_to_peak = float(np.sum(self.new2[:k_peak + 1]))
        cum_total_to_peak = cum1_to_peak + cum2_to_peak

        t_peak = float(self.t[k_peak])
        I1_peak = float(self.I1[k_peak])
        frac_total_to_peak = cum_total_to_peak / float(self.N)

        return frac_total_to_peak
    
    def i1_near_extinction_fraction(self, eps=10.0, t_end=500.0):
        """
        Choose a finite horizon t_end and define k_ext using ONLY that range.

        k_ext is set based on the last time I1 >= eps in [t_peak, t_end].
        If I1 never drops below eps permanently within that window, this will reflect that.

        Returns:
        cum_events_per_capita up to k_ext (can exceed 1 if reinfections exist), or None
        if t_end is before the peak / invalid.
        """
        if self.t is None:
            self.run()

        # indices up to chosen horizon
        k_end = int(np.searchsorted(self.t, t_end, side="right")) - 1
        if k_end < 0:
            return None

        I1 = self.I1
        k_peak = int(np.argmax(I1))

        # must have a window after the peak
        if k_end <= k_peak:
            return None

        # last time I1 >= eps between peak and t_end (inclusive)
        seg = I1[k_peak:k_end + 1]
        idx = np.where(seg >= eps)[0]

        if idx.size == 0:
            # never above eps after peak within the window -> extinction effectively at peak
            k_ext = k_peak
        else:
            k_last_high = k_peak + int(idx[-1])
            # choose ONE of these definitions:

            k_ext = k_last_high          # "last time it's still above eps"
            # k_ext = k_last_high + 1    # "first time after the last above-eps point"

            if k_ext > k_end:
                return None

        cum_total = float(np.sum(self.new1[:k_ext + 1]) + np.sum(self.new2[:k_ext + 1]))
        return cum_total / float(self.N)


