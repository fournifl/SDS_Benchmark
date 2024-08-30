import argparse
import pdb
import numpy as np
from pathlib import Path
from dataclasses import dataclass, fields

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union


@dataclass
class RollingMeanStdVarNames:
    """Defines derived variable names from base variable name
    for rolling mean std filter function.
    """

    var: str
    minmax_outliers: str = "minmax_outliers"
    mean: str = "mean"
    std: str = "std"
    low: str = "low"
    high: str = "high"
    meanstd_outliers: str = "meanstd_outliers"
    valid: str = "valid"

    def __post_init__(self):
        for field in fields(self):
            if field.name == "var":
                pass
            else:
                value = getattr(self, field.name)
                new_value = "_".join([self.var, value])
                setattr(self, field.name, new_value)
        return self

    def to_list(self):
        fields_list = []
        for field in fields(self):
            fields_list.append(getattr(self, field.name))
        return fields_list


def filter_df_rolling_mean_std(
    df: pd.DataFrame,
    column: str,
    window: Union[str, int] = "7D",
    n: float = 1.5,
    mask_column: str = None,
    limits: Tuple[float, float] = None,
    out_name: str = None,
    inplace=False,
) -> pd.DataFrame:
    """Filter time series `column` of dataframe `df` with a rolling (mean +/- std)
    filter.

    Filter `df.column` using a rolling window of size `window`.
    The rolling mean and standard deviation are computed over `column`, and all data
    points outside of `(mean +/- (n * std))` are identified as outliers.

    Arguments:
        - df : input dataframe. Must have an index column with datatype datetime

        - column : name of the df column to be filtered

        - window : defines the rolling windows as a fixed number of observations
        (set an integer) or a variable number of observations based on an offset
        (ex : '12D' for twelve days, see pandas doc on rolling windows and timedeltas)

        - n : defines the width of the interval above and below the mean value
        (the n in `mean +/- (n * std)`) cant be an  int or float

        - mask_column : name of the df column to be used as a pre fitler boolean mask

        - limits : (minimum, maximum) allowed values for pre filter

        - out_name : rename the input column

    Returns:
        df : the input df with new columns :
            - `column_valid` boolean column, set to True for valid points
                (lying in (min +/- n*std)) and False otherwise.

            - the different columns used to compute the final filter ( mean, std, etc.).
                These columns are named as `column_mean`, `column_std`, etc.
    """

    if column not in df.columns:
        raise ValueError(f"Dataframe does not contain Column {column}")

    if not inplace:
        df: pd.DataFrame = df.copy()

    if out_name:
        var_name = out_name
        df[var_name] = df[column]
    else:
        var_name = column

    names = RollingMeanStdVarNames(var_name)

    if mask_column:
        df[names.var] = df[names.var].mask(~df[mask_column])
    if limits:
        min_threshold = limits[0] if limits[0] is not None else df[names.var].min()
        max_threshold = limits[1] if limits[1] is not None else df[names.var].max()

        if max_threshold < min_threshold:
            min_threshold, max_threshold = max_threshold, min_threshold
        df[names.minmax_outliers] = (df[names.var] < min_threshold) | (
            df[names.var] > max_threshold
        )
        df[names.valid] = ~df[names.minmax_outliers]
    else:
        df[names.minmax_outliers] = False
        df[names.valid] = True

    rolling = (
        df[names.var]
        .mask(df[names.minmax_outliers])
        .rolling(window=window, center=True)
    )
    df[names.mean], df[names.std] = rolling.mean(), rolling.std()
    df[names.low] = df[names.mean] - (n * df[names.std])
    df[names.high] = df[names.mean] + (n * df[names.std])
    df[names.meanstd_outliers] = (df[names.var] > df[names.high]) | (
        df[names.var] < df[names.low]
    )

    df[names.valid] = df[names.valid] & ~df[names.meanstd_outliers]

    if mask_column:
        df[names.valid] = df[names.valid] & df[mask_column]

    if inplace:
        return None
    else:
        return df


def filter_df_rolling_mean_std_two_pass(
    df, column, limits=None, window="7D", n1=1.5, n2=1.5, inplace=True
):
    """Apply twice the `filter_df_rolling_mean_std` function on a dataframe's column."""
    if inplace:
        filter_df_rolling_mean_std(
            df, column=column, limits=limits, window=window, n=n1, inplace=True
        )
        filter_df_rolling_mean_std(
            df,
            column=column,
            mask_column=column + "_valid",
            out_name=column + "2",
            window=window,
            n=n2,
            inplace=True,
        )
        return None
    else:
        dataframe = df.copy()
        filter_df_rolling_mean_std(
            dataframe, column=column, limits=limits, window=window, n=n1, inplace=True
        )
        filter_df_rolling_mean_std(
            dataframe,
            column=column,
            mask_column=column + "_valid",
            out_name=column + "2",
            window=window,
            n=n2,
            inplace=True,
        )
        return dataframe
