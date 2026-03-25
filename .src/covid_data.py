"""
Generates a class containing the relevant covid data in a pd.dataframe and a plotting function.
CovidData().data is the dataframe. CovidData().plot() does the plotting.
This is not used in the final report, however is once more kept.
"""
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# If using other datasets would turn this from dataclass to just hving an init.
@dataclass
class CovidData:
    filename: str = "covid_data.xlsx"
    sheet_name: str = "1a"
    skiprows: Optional[int] = None

    # you can change these if your sheet differs
    date_col: int = 0          # column index in Excel for the date-range strings
    infected_col: int = 1      # column index in Excel for "% infected" (or whatever value)

    df: Optional[pd.DataFrame] = None

    @staticmethod
    def _parse_mid_date(range_str: Union[str, float, None]) -> pd.Timestamp:
        """
        Convert a string like "27 April 2020 to 10 May 2020" to the midpoint date.
        Returns pd.NaT if it can't parse.
        """
        if range_str is None or (isinstance(range_str, float) and np.isnan(range_str)):
            return pd.NaT

        s = str(range_str).strip()
        if not s:
            return pd.NaT

        # Split on " to " (case-insensitive, tolerant of extra spaces)
        parts = re.split(r"\s+to\s+", s, flags=re.IGNORECASE)
        if len(parts) != 2:
            return pd.NaT

        start_raw, end_raw = parts[0].strip(), parts[1].strip()

        # Let pandas parse "27 April 2020"
        start = pd.to_datetime(start_raw, errors="coerce", dayfirst=True)
        end = pd.to_datetime(end_raw, errors="coerce", dayfirst=True)

        if pd.isna(start) or pd.isna(end):
            return pd.NaT

        # Midpoint in time
        return start + (end - start) / 2
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Option 1: Load data and return a dataframe with:
          - mid_date (datetime)
          - infected (float)
        """
        raw = pd.read_excel(
            self.filename,
            sheet_name=self.sheet_name,
            usecols=[self.date_col, self.infected_col],
            skiprows=self.skiprows,
            header=None,
        )
        raw.columns = ["date_range", "infected"]

        out = pd.DataFrame(
            {
                "mid_date": raw["date_range"].apply(self._parse_mid_date),
                "infected": pd.to_numeric(raw["infected"], errors="coerce"),
            }
        ).dropna(subset=["mid_date", "infected"])

        out = out.sort_values("mid_date").reset_index(drop=True)
        self.df = out
        return out
    

    def estimate_beta_per_step(
        self,
        gamma,
        dt=7.0,
    ):
        """
        Estimate R_eff(t) and beta(t) using
            beta = d/dt ln I + gamma

        Parameters
        ----------
        gamma : float
            Recovery rate (e.g. 1/7 per day)
        dt : float
            Time step between data points (in days)

        Returns
        -------
        R_eff : np.ndarray
            Effective reproduction number at each timestep
        beta : np.ndarray
            Effective transmission rate at each timestep
        """
        I = np.asarray(self.data['infected'], dtype=float)

        n = len(I)
        R_eff = np.full(n - 1, np.nan)
        beta = np.full(n - 1, np.nan)

        for t in range(n - 1):
            if I[t] <= 0 or I[t + 1] <= 0:
                continue

            dlogI_dt = (np.log(I[t + 1]) - np.log(I[t])) / dt
            beta[t] = dlogI_dt + gamma
            R_eff[t] = beta[t] / gamma

        return R_eff, beta

    
    def plot(self, use_dates: bool = True) -> None:
        """
        Option 2: Plotting function.

        - If use_dates=True: x-axis is mid_date
        - Else: x-axis is index (0..N-1)
        """
        if self.df is None:
            self.data

        assert self.df is not None  # for type checkers
        x = self.df["mid_date"] if use_dates else np.arange(len(self.df))
        y = self.df["infected"]

        plt.figure(figsize=(7, 4))
        plt.plot(x, y, label="% infected")
        plt.axvline(pd.Timestamp("2022-02-24"))       
        plt.xlabel("Mid date" if use_dates else "Index (time step)")
        plt.ylabel("% infected")
        plt.title("COVID data")
        plt.legend()
        plt.tight_layout()
        plt.show()
