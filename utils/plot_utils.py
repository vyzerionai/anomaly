"""Plotting utility for anomaly detection."""
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List
from . import sample_utils


def get_palettes(desired_len, palettes: List[str] = None):
    """Return a set of palettes
    Args:
        desired_len: the number of palettes options desired
        palettes: Optional set of palettes

    Returns:
        An array of palettes for plotting
    """
    if palettes is None:
        palettes = ['PuBu', 'Reds', 'Greens', 'Blues', 'Oranges', 'PuRd',
                    'Wistia', 'Purples', 'pink_r', ]

    current_len = len(palettes)

    if desired_len > current_len:
        palettes.extend(np.random.choice(palettes, size=desired_len - current_len,
                                         replace=True if (desired_len - current_len) > current_len else False))

    return np.random.choice(palettes, desired_len, replace=False)


def plot_all_feature_spreads(feature_lists, df_consolidated: pd.DataFrame,
                             observations: pd.DataFrame, palettes=None,
                             png_dir=None, show_plot=True):
    """
    Create boxplots of features that are aggregated together
    Args:
        feature_lists: lists of features to plot together
        df_consolidated: dataframe with a sample of the training or consolidated data
        observations: dataframe of the observations from the selected flight
        palettes: optional set of palettes to plot the different feature plots
        png_dir: optional directory to save all the feature spreads as png
        show_plot: option of whether or not to display the plots generated
    Returns: None
    """
    palettes = get_palettes(len(feature_lists), palettes)

    for color, selected_features in zip(palettes, feature_lists):
        df_feature = create_feature_df(df_consolidated, observations,
                                       selected_features, normalize_data=False)
        plot_feature_spread(df_feature, palette=color, png_dir=png_dir,
                            show_plot=show_plot)


def create_feature_df(df_consolidated: pd.DataFrame, df_flight: pd.DataFrame,
                      features: List[str], normalize_data=True) -> pd.DataFrame:
    """
    Creating a merged df to compare baseline and values based on features
    Args:
        df_consolidated: dataframe with a sample of the training or consolidated data
        df_flight: dataframe of a specific flight
        features: list of features
        normalize_data: whether to normalize the data
    Returns:
        A dataframe containing each feature with value and baseline.
    """
    df_baseline = df_consolidated[features].sample(n=min(200000, len(df_consolidated)))

    df_flight = df_flight[features].copy()

    if normalize_data:
        normalization_data = sample_utils.get_normalization_info(df_baseline)

        df_baseline = sample_utils.normalize(df_baseline, normalization_data)
        df_flight = sample_utils.normalize(df_flight, normalization_data)

    df_baseline['baseline'] = 'baseline'
    df_flight['baseline'] = 'flight'

    df_merged = pd.concat([df_baseline, df_flight], axis=0)

    feature_dfs = []
    for feature in features:
        feature_df = df_merged[[feature, 'baseline']].copy()
        feature_df['feature'] = feature
        feature_df.columns = ['feature_value', 'is_baseline', 'feature_name']
        feature_dfs.append(feature_df)

    return pd.concat(feature_dfs, axis=0, ignore_index=True)


def plot_feature_spread(df_feature, palette="Greens", png_dir=None, show_plot=True):
    """Plots a boxplot for each feature in a flight."""
    nfeatures = len(df_feature["feature_name"].unique())
    _ = plt.figure(figsize=(20, nfeatures))
    sns.set(style="whitegrid")

    ax = sns.boxplot(
        data=df_feature,
        y="feature_name",
        x="feature_value",
        hue="is_baseline",
        palette=sns.color_palette(palette, 2),
    )
    ax.grid(True)

    ax.set_title("Feature Spread comparison", fontsize=20)
    plt.legend(fontsize=16)

    # Saving as png
    if png_dir is not None:
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)

        # Identifying the png by the first feature in the feature spread
        file_name = f'%s_Feature_spread.png' % df_feature['feature_name'][0]
        plt.savefig(os.path.join(png_dir, file_name),
                    dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()


def plot_attribution(
        df_attribution: pd.DataFrame, anomaly_score: float, engine_sn,
        flight_sn, timestamp, png_dir=None, show_plot=True) -> None:
    """Plots the attribution as a pie chart.

    The center contains the anomaly score. The wedges are ordered clockwise
    from the most blame to the least. The percentages are the normalized
    percentages (Blame(d) / Sum of Blames). The values outside the wedges
    indicate the observed value, and the expected value in parentheses.

    Args:
        df_attribution: dataframe with observed_value, expected_value, and
        attribution for each dimension.
        anomaly_score: score ranging between Normal (1) and Anomalous (0).
        engine_sn: engine serial number
        flight_sn: flight serial number
        timestamp: timestamp of the observed point
        png_dir: optional directory to save the pie chart as a png file
        show_plot: whether to display plots
    Returns: None
    """
    df_attribution = df_attribution.sort_values(by="attribution", ascending=False)
    norm = plt.Normalize()
    names = []
    sizes = []
    sum_big = 0
    for fn, row in df_attribution.iterrows():
        # Only show the dimensions with a blame > 5%.
        if row.attribution > 0.03:
            if fn.endswith("oscillation"):
                names.append("%s\n" % fn)
            else:
                names.append("%s\n%3.1f (%3.1f)" % (
                    fn, row.observed_value, row.expected_value))
            wedge_size = int(100 * row.attribution)
            sum_big += wedge_size
            sizes.append(wedge_size)
    names.append("other")
    sizes.append(int(100 - sum_big))

    # Create a circle for the center of the plot
    num_p_score_steps = 100
    center_color_index = int(num_p_score_steps * anomaly_score)
    my_circle = plt.Circle(
        (0, 0),
        0.45,
        facecolor=plt.cm.RdYlGn(norm(range(num_p_score_steps + 1)))[center_color_index],
        edgecolor="white",
        linewidth=3,
    )

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    center_text = "%.2f" % anomaly_score
    if (center_color_index < 20) or (center_color_index > 80):
        text_color = "white"
    else:
        text_color = "black"
    ax.text(
        0,
        0,
        center_text,
        fontsize=28,
        horizontalalignment="center",
        color=text_color,
        weight="bold",
    )

    # Custom colors --> colors will cycle
    norm = plt.Normalize()
    # Choose nine colors to cycle through to show contrast between slices.
    pie_plot = plt.pie(
        sizes,
        labels=names,
        colors=plt.cm.RdYlBu(norm(range(9)), alpha=0.6),
        startangle=90,
        counterclock=False,
        autopct="%1.0f%%",
        pctdistance=0.70,
        textprops=dict(color="black", weight="bold", fontsize=28),
    )

    plt.title("%s %s %s" % (engine_sn, flight_sn, timestamp))

    for lab in pie_plot[1]:
        lab.set_fontsize(28)
    p = plt.gcf()
    p.gca().add_artist(my_circle)

    if png_dir is not None:
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
        file_name = f"Engine_%s_Flight_%s_timestamp_%s_attribution_pie_chart.png" % (
            engine_sn, flight_sn, timestamp)
        plt.savefig(os.path.join(png_dir, file_name),
                    dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()


def plot_gradient_series(df_grad: pd.DataFrame, delta: np.array,
                         show_plot=True) -> None:
    """Plots the gradient for each feature from baseline to target point."""
    fig, ax = plt.subplots()
    fig.set_figheight(12)
    fig.set_figwidth(15)
    n_points = len(df_grad)
    colors = sns.color_palette("rainbow", df_grad.shape[1])
    for ix, field_name in enumerate(df_grad):
        series_color = colors[ix]
        ig_series = (df_grad[field_name].cumsum() / float(n_points)) * delta[field_name]
        ax.plot(
            df_grad.index,
            ig_series,
            linewidth=3.0,
            linestyle="-",
            marker=None,
            color=series_color,
        )

    ax.grid(linestyle="-", linewidth="0.5", color="darkgray")

    legend = plt.legend(
        loc="upper left",
        shadow=False,
        fontsize="16",
        bbox_to_anchor=(1, 1),
        labels=list(df_grad),
    )
    legend.get_frame().set_facecolor("white")
    plt.ylabel("Cumulative Gradients")

    for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
    ):
        item.set_fontsize(24)

    for sp in ax.spines:
        ax.spines[sp].set_color("black")
    ax.set_facecolor("white")

    if show_plot:
        plt.show()
        plt.grid(True)


def get_single_ad_timeseries(ad, observations: pd.DataFrame) -> pd.DataFrame:
    """Returns the single timeseries of predictions."""

    if 'class_label' in observations.columns:
        observations = observations.drop(columns=['class_label'])
    return ad.predict(observations)['class_prob']


def plot_variable_timeseries(observations: pd.DataFrame, variable_name: str,
                             label: str,
                             predictions: Optional[pd.Series] = None,
                             anomaly_smoothing_kernel: int = 15,
                             timeseries_dir: str = None,
                             ad_name: str = 'AD', show_plot=True):
    """Plots each variable as a timeseries with anomaly detection score.

    Args:
        observations: Dataframe with columns as features.
        variable_name: Column name to plot (must be a col in the observations).
        label: Added to the title of the chart.
        predictions: Data series containing the anomaly detector's predictions.
        anomaly_smoothing_kernel: window size to smooth the anomaly score.
        timeseries_dir: directory to save timeseries as a png
        ad_name: name of the AD
        show_plot: whether to display plots
    """
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    fig.set_figheight(15)
    fig.set_figwidth(50)
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)
    ax2.tick_params(axis="y", labelsize=24)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Anomaly Score", color="red", fontsize=14)
    ax.xaxis.set_minor_locator(mdates.MinuteLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    timestamps = [ii[2] for ii in observations.index]

    ax.plot(
        timestamps,
        observations[variable_name],
        label=variable_name,
        linewidth=2.0,
        marker=None,
        color="green",
    )

    if 'class_label' in observations.columns:
        anomalous_observations = observations[observations['class_label'] == 0]
        anomalous_timestamps = [ii[2] for ii in anomalous_observations.index]

        ax.plot(anomalous_timestamps, anomalous_observations[variable_name],
                label=variable_name, linewidth=1.0, marker='.', color='red')

    if 'class_prob' in observations.columns or predictions is not None:
        if predictions is not None:
            ad_predictions = predictions
        else:
            ad_predictions = observations['class_prob']
        ad_predictions = 1 - ad_predictions

        kernel_size = anomaly_smoothing_kernel
        kernel = np.ones(kernel_size) / kernel_size
        ad_smoothed = np.convolve(ad_predictions, kernel, mode="same")

        ax2.plot(
            timestamps,
            ad_predictions,
            label=ad_name,
            linewidth=1.0,
            marker=None,
            color="lightgray",
        )
        ax2.plot(
            timestamps,
            ad_smoothed,
            label=ad_name,
            linewidth=2.0,
            marker=None,
            color="orange",
        )

    for tick in ax.get_xticklabels():
        tick.set_rotation(0)

    plt.title("%s timeseries %s" % (variable_name, label), fontsize=24)
    plt.legend(loc="lower left", fontsize=16)
    plt.grid(True)

    if timeseries_dir is not None:
        if not os.path.exists(timeseries_dir):
            os.makedirs(timeseries_dir)
        file_name = f"%s-%s_timeseries_plot.png" % (label, variable_name)
        plt.savefig(os.path.join(timeseries_dir, file_name),
                    dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()


def plot_pair_plots(observations):
    """Plots scatter plots of the data."""
    variables = list(observations)
    if "class_label" in variables:
        variables.remove("class_label")
        cell_count = len(variables) - 1
    fig, ax = plt.subplots(nrows=cell_count, ncols=cell_count)

    if "class_label" in observations.columns:
        anomalous_observations = observations[observations["class_label"] == 0]
    else:
        anomalous_observations = None

    fig.set_figheight(10 * cell_count)
    fig.set_figwidth(10 * cell_count)
    rx = 1
    for row in ax:
        y_label = variables[rx]
        cx = 0
        for col in row:
            if rx > cx:
                x_label = variables[cx]

                col.plot(
                    observations[x_label],
                    observations[y_label],
                    ".",
                    markersize=4.0,
                    color="green",
                    alpha=0.8,
                )

                if anomalous_observations is not None:
                    col.plot(
                        anomalous_observations[x_label],
                        anomalous_observations[y_label],
                        ".",
                        markersize=5.0,
                        color="red",
                        alpha=0.8,
                    )

                col.set_xlabel(x_label)
                col.set_ylabel(y_label)
            else:
                col.axis("off")
            cx = cx + 1
        rx = rx + 1
