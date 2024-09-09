"""Utilities to to generate or modify data samples."""
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf


class Variable(object):
    def __init__(self, index, name, mean, std, min=None, max=None):
        self.index = index
        self.mean = mean
        self.name = name
        self.std = std
        self.min = min
        self.max = max


NormalizationInfo = Dict[str, Variable]


def get_minmax_normalization_info(df: pd.DataFrame) -> Dict[str, Variable]:
    variables = {}
    for column in df:
        if not np.issubdtype(df[column].dtype, np.number):
            raise ValueError("The feature column %s is not numeric." % column)

        if column.endswith("_validity"):
            vmin = 0
            vmax = 0
        else:
            vmin = df[column].min()
            vmax = df[column].max()

        variable = Variable(
            index=df.columns.get_loc(column),
            name=column,
            mean=None,
            std=None,
            min=vmin,
            max=vmax,
        )
        variables[column] = variable
    return variables


def get_normalization_info(df: pd.DataFrame) -> Dict[str, Variable]:
    """Computes means, standard deviation to normalize a data frame.

    Any variable xxxx_validity is considered a boolean validity indicator
    for variable xxxx, and will not be normalized. A value of 1
    indicates the value xxxx is valid, and 0 indicates xxx is invalid.

    Args:
      df: Pandas dataframe with numeric feature data.

    Returns:
      A dict with Variable.name, Variable.
    """
    variables = {}
    for column in df:
        if not np.issubdtype(df[column].dtype, np.number):
            raise ValueError("The feature column %s is not numeric." % column)

        if column.startswith("aux"):
            print("Auxiliary column %s is not normalized." % column)
            vmean = 0.0
            vstd = 1.0
        else:
            vmean = df[column].mean()
            vstd = df[column].std()

        variable = Variable(
            index=df.columns.get_loc(column), name=column, mean=vmean, std=vstd
        )
        variables[column] = variable
    return variables


def get_column_order(normalization_info: Dict[str, Variable]) -> List[str]:
    """Returns a list of column names, as strings, in model order."""
    return [
        var.name
        for var in sorted(normalization_info.values(), key=lambda var: var.index)
    ]


def normalize(
    df: pd.DataFrame, normalization_info: Dict[str, Variable]
) -> pd.DataFrame:
    """Normalizes an input Dataframe of features.

    Args:
      df: Pandas DataFrame of M rows with N real-valued features
      normalization_info: dict of name, variable types containing mean, and std.

    Returns:
      Pandas M x N DataFrame with normalized features.
    """

    df_norm = pd.DataFrame()
    not_norm_cols = list(set(df.columns) - set(get_column_order(normalization_info)))
    for column in get_column_order(normalization_info):
        if (
            normalization_info[column].std is not None
            and normalization_info[column].mean is not None
        ):
            if normalization_info[column].std == 0.0:
                df_norm[column] = 0.0
            else:
                df_norm[column] = (
                    df[column] - normalization_info[column].mean
                ) / normalization_info[column].std

        elif (
            normalization_info[column].min is not None
            and normalization_info[column].max is not None
        ):
            df_norm[column] = (df[column] - normalization_info[column].min) / (
                normalization_info[column].max - normalization_info[column].min
            )

        else:
            raise ValueError("Normalization information is invalid.")

    return pd.concat([df_norm, df[not_norm_cols]], axis=1)


def denormalize(
    df_norm: pd.DataFrame, normalization_info: Dict[str, Variable]
) -> pd.DataFrame:
    """Reverts normalization an input Dataframe of features.

    Args:
      df_norm: Pandas DataFrame of M rows with N real-valued normalized features
      normalization_info: dict of name, variable types containing mean, and std.

    Returns:
      Pandas M x N DataFrame with denormalized features.
    """
    df = pd.DataFrame()
    not_norm_cols = list(set(df.columns) - set(get_column_order(normalization_info)))
    for column in get_column_order(normalization_info):
        if (
            normalization_info[column].std is not None
            and normalization_info[column].mean is not None
        ):
            if normalization_info[column].std == 0.0:
                df[column] = normalization_info[column].mean
            else:
                df[column] = (
                    df_norm[column] * normalization_info[column].std
                    + normalization_info[column].mean
                )
        elif (
            normalization_info[column].min is not None
            and normalization_info[column].max is not None
        ):
            df[column] = (
                df_norm[column]
                * (normalization_info[column].max - normalization_info[column].min)
                + normalization_info[column].min
            )

        else:
            raise ValueError("Normalization information is invalid.")
    return pd.concat([df, df_norm[not_norm_cols]], axis=1)


def write_normalization_info(normalization_info: Dict[str, Variable], filename: str):
    """Writes variable normalization info to CSV."""

    def to_df(normalization_info):
        df = pd.DataFrame(columns=["index", "mean", "std"])
        for variable in normalization_info:
            df.loc[variable] = [
                normalization_info[variable].index,
                normalization_info[variable].mean,
                normalization_info[variable].std,
            ]
        return df

    with tf.io.gfile.GFile(filename, "w") as csv_file:
        to_df(normalization_info).to_csv(csv_file, sep="\t")


def read_normalization_info(filename: str) -> Dict[str, Variable]:
    """Reads variable normalization info from CSV."""

    def from_df(df):
        normalization_info = {}
        for name, row in df.iterrows():
            normalization_info[name] = Variable(
                row["index"], name, row["mean"], row["std"]
            )
        return normalization_info

    normalization_info = {}
    if not tf.io.gfile.exists(filename):
        raise AssertionError("{} does not exist".format(filename))
    with tf.io.gfile.GFile(filename, "r") as csv_file:
        df = pd.read_csv(csv_file, header=0, index_col=0, sep="\t")
        normalization_info = from_df(df)
    return normalization_info

