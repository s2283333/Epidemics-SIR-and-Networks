"""
Generates a class containing the relevant covid data in a pd.dataframe and a plotting function.
CovidData().data is the dataframe. CovidData().plot() does the plotting.
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
    def estimate_beta(
        self,
        start,
        end,
        gamma,
            ):
        df = self.data if self.df is None else self.df

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        win = df[(df["mid_date"] >= start) & (df["mid_date"] <= end)]
        win = win[win["infected"] > 0].sort_values("mid_date")

        t = (win["mid_date"] - win["mid_date"].iloc[0]).dt.days.to_numpy()
        I = win["infected"].to_numpy()

        r, _ = np.polyfit(t, np.log(I), 1)

        beta = r + gamma
        return beta
    
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

#TODO:
# 1. Look at cutoff of regulations -> oscillations.
# 2. match to network/SIRS.
CovidData(sheet_name='1a').plot()
beta = CovidData().estimate_beta(
    start="2021-11-30",
    end="2021-12-29",
    gamma=1/5,
)
print(beta)
