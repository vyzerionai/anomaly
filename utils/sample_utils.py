"""Utilities to to generate or modify data samples."""

import collections
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging
from keras.utils import to_categorical


class Variable(object):
    def __init__(self, index, name, mean, std, min=None, max=None):
        self.index = index
        self.mean = mean
        self.name = name
        self.std = std
        self.min = min
        self.max = max


NormalizationInfo = Dict[str, Variable]


def get_categoricals(sample: pd.DataFrame) -> Mapping[str, Sequence[tuple[str, int]]]:
    columns = sample.columns
    categoricals = collections.defaultdict(list)
    for col in columns:
        count = sample[col].sum()

        if col.startswith("aux_cat"):
            splits = col.split("_")
            categoricals[splits[2]].append((splits[3], count))
    return categoricals


def sample_categorical(
    cat_name: str,
    class_counts: Sequence[tuple[str, int]],
    n_points: int,
    equal_odds=False,
    min_prob=0.05,
):
    if equal_odds:
        p_class = [1.0 / len(class_counts) for _ in class_counts]
    else:
        total_count = float(np.sum([cat[1] for cat in class_counts]))
        p_class = [cat[1] / total_count for cat in class_counts]
        p_class = [min_prob + p for p in p_class]
        p_class /= np.sum(p_class)

    index_array = np.random.choice(len(class_counts), n_points, p=p_class)

    one_hot = to_categorical(index_array, len(class_counts))

    columns = ["aux_cat_%s_%s" % (cat_name, cat[0]) for cat in class_counts]

    return pd.DataFrame(one_hot, columns=columns)


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
            logging.info("Auxiliary column %s is not normalized." % column)
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

    if not tf.io.gfile.exists(filename):
        raise AssertionError("{} does not exist".format(filename))
    with tf.io.gfile.GFile(filename, "r") as csv_file:
        df = pd.read_csv(csv_file, header=0, index_col=0, sep="\t")
        normalization_info = from_df(df)
    return normalization_info


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


def get_neg_sample(
    pos_sample: pd.DataFrame,
    n_points: int,
    do_permute: bool = False,
    delta: float = 0.0,
) -> pd.DataFrame:
    """Creates a negative sample from the cuboid bounded by +/- delta.

    Where, [min - delta, max + delta] for each of the dimensions.
    If do_permute, then rather than uniformly sampling, simply
    randomly permute each dimension independently.
    The positive sample, pos_sample is a pandas DF that has a column
    labeled 'class_label' where 1.0 indicates Normal, and
    0.0 indicates anomalous.

    Args:
      pos_sample: DF with numeric dimensions
      n_points: number points to be returned
      do_permute: permute or sample
      delta: fraction of [max - min] to extend the sampling.

    Returns:
      A dataframe  with the same number of columns, and a label column
      'class_label' where every point is 0.
    """
    df_neg = pd.DataFrame()

    pos_sample_n = pos_sample.sample(n=n_points, replace=True)

    for field_name in list(pos_sample):
        if field_name == "class_label":
            continue

        if field_name.startswith("aux_cat"):
            continue

        if do_permute:
            df_neg[field_name] = np.random.permutation(
                np.array(pos_sample_n[field_name])
            )

        # If all the points are integers, then sample from integers only.

        if all(v.is_integer() for v in pos_sample[field_name]):
            integer_range = list(set(pos_sample[field_name]))
            df_neg[field_name] = np.random.choice(
                integer_range, size=n_points, replace=True
            ).astype(np.float32)

        else:
            low_val = min(pos_sample[field_name])
            high_val = max(pos_sample[field_name])
            delta_val = high_val - low_val
            df_neg[field_name] = np.random.uniform(
                low=low_val - delta * delta_val,
                high=high_val + delta * delta_val,
                size=n_points,
            )

    categoricals = get_categoricals(pos_sample)

    if categoricals:
        cat_dfs = []
        for cat in categoricals:
            cat_df = sample_categorical(
                cat, categoricals[cat], n_points=n_points, equal_odds=False
            )
            cat_dfs.append(cat_df)

        categorical_df = pd.concat(cat_dfs, axis=1)
        categorical_df.index = df_neg.index

        df_neg = pd.concat([df_neg, categorical_df], axis=1)

    df_neg["class_label"] = [0 for _ in range(n_points)]
    return df_neg[pos_sample.columns]


def apply_negative_sample(
    positive_sample: pd.DataFrame,
    sample_ratio: float,
    sample_delta: float,
    do_permute=False,
) -> pd.DataFrame:
    """Returns a dataset with negative and positive sample.

    Args:
        positive_sample: actual, observed sample where each col is a feature.
        sample_ratio: the desired ratio of negative to positive points
        sample_delta: the extension beyond observed limits to bound the neg sample
        do_permute: permute or sample
    Returns:
        DataFrame with features + class label, with 1 being observed and 0 negative.
    """

    positive_sample["class_label"] = 1
    n_neg_points = int(len(positive_sample) * sample_ratio)

    negative_sample = get_neg_sample(
        positive_sample, n_neg_points, do_permute=do_permute, delta=sample_delta
    )
    training_sample = pd.concat(
        [positive_sample, negative_sample], ignore_index=True, sort=True
    )
    return training_sample.reindex(np.random.permutation(training_sample.index))
