import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Optional, Dict
from sklearn.metrics import confusion_matrix

def plot_events(HS_error, TO_error, path: Optional[str] = None, return_plot: bool = False):
    cm = 1 / 2.54

    hs_cols = ['LF_HS', 'RF_HS']
    to_cols = ['LF_TO', 'RF_TO']

    fig, axs = plt.subplots(1, 2, figsize=(36 * cm, 16 * cm), sharey=True)

    sns.boxplot(ax=axs[0], data=HS_error, y="time_error", x="event", order=hs_cols, showfliers=True,
                flierprops=dict(marker='o', ms=2, mec=(0, 0, 0), alpha=0.4, mfc='none'), palette='pastel')
    sns.despine(bottom=True)
    axs[0].set_xlabel("")
    axs[0].set_xticklabels([s.replace("_HS", '') for s in hs_cols])
    axs[0].set_ylabel('time error (in s)')  # set ylabel
    axs[0].yaxis.set_minor_locator(plt.MultipleLocator(10))
    axs[0].grid(visible=True, which="major", axis="y", c=(0, 0, 0), alpha=0.1, ls=":")
    axs[0].set_title("Heel Strike")
    axs[0].tick_params(axis="x", which="both", length=0)

    sns.boxplot(ax=axs[1], data=TO_error, y="time_error", x="event", order=to_cols, showfliers=True,
                flierprops=dict(marker='o', ms=2, mec=(0, 0, 0), alpha=0.4, mfc='none'), palette='pastel')
    sns.despine(bottom=True)
    axs[1].set_xlabel("")
    axs[1].set_ylabel("")
    axs[1].set_xticklabels([s.replace("_TO", '') for s in to_cols])
    axs[1].yaxis.set_minor_locator(plt.MultipleLocator(10))
    axs[1].grid(visible=True, which="major", axis="y", c=(0, 0, 0), alpha=0.1, ls=":")
    axs[1].set_title("Toe Off")
    axs[1].tick_params(axis="x", which="both", length=0)

    plt.tight_layout()

    if return_plot:
        return fig

    else:
        filepath = os.path.join(path, 'event_metrics' + ".png")
        plt.savefig(filepath, format="png", bbox_inches="tight")
        plt.close()

def bland_altman_plot(m1, m2, sd_limit=1.96, ax=None, annotate=True,
                        scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None):

    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError("Matplotlib is not found.")
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kwds)
    ax.axhline(mean_diff, **mean_line_kwds)

    # Annotate mean line with mean difference.
    if annotate:
        ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                    xy=(0.99, 0.5),
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=14,
                    xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        if annotate:
            range_lim = (0.20 - (-0.20))
            low_lim = (lower - (-0.20))/(0.20 - (-0.20))
            high_lim = (upper - (-0.20))/(0.20 - (-0.20))
            ax.annotate(f'\N{MINUS SIGN}{sd_limit} SD: {lower:0.2g}',
                        xy=(0.99, low_lim-0.02*range_lim), # (0.99, 0.07),
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=14,
                        xycoords='axes fraction')
            ax.annotate(f'+{sd_limit} SD: {upper:0.2g}',
                        xy=(0.99, high_lim+0.02*range_lim), # (0.99, 0.92),
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        fontsize=14,
                        xycoords='axes fraction')
    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    ax.set_ylabel('Difference', fontsize=15)
    ax.set_xlabel('Means', fontsize=15)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    return fig

def plot_parameters(gait_parameters: pd.DataFrame, plot_undetected: bool = False, path: Optional[str] = None):

    colors = ['red', 'green', 'blue']
    if plot_undetected:
        for foot in ["LF", "RF"]:
            foot_parameters = gait_parameters[gait_parameters['foot'] == foot]

            cm = 1 / 2.54
            fig, axs = plt.subplots(1, 3, figsize=(43 * cm, 14 * cm))

            for p, param_name in enumerate(foot_parameters.parameter.unique()):
                param_time = foot_parameters[foot_parameters['parameter'] == param_name]
                param_time['detected'] = ~param_time['pred_value'].isnull()

                sns.histplot(param_time, x='true_value', hue='detected', ax=axs[p],
                             bins=np.linspace(0.25, 1.5, 50), color=colors[p],
                             stat='probability', common_norm=False)

                axs[p].set_title(param_name, size=18)
                axs[p].set_xlabel("Duration (in s)")
                axs[p].set_ylabel("Number of Samples")
                axs[p].xaxis.set_minor_locator(plt.MultipleLocator(0.1))

            plt.tight_layout()
            plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=16)

            filepath = os.path.join(path, foot + '_detected_parameters' + '.png')
            plt.savefig(filepath, format="png", bbox_inches="tight")
            plt.close()

    gait_parameters = gait_parameters.dropna()

    for foot in ["LF", "RF"]:
        foot_parameters = gait_parameters[gait_parameters['foot'] == foot]

        cm = 1 / 2.54
        fig, axs = plt.subplots(1, 3, figsize=(43 * cm, 14 * cm), sharey=True)

        for p, param_name in enumerate(foot_parameters.parameter.unique()):
            param_time = foot_parameters[foot_parameters['parameter'] == param_name]

            bland_altman_plot(param_time.true_value, param_time.pred_value, ax=axs[p],
                              scatter_kwds=dict(c="w", alpha=0.01))

            bland_altman_plot(param_time.true_value, param_time.pred_value, ax=axs[p],
                              annotate=False, mean_line_kwds=dict(ls="none"),
                              limit_lines_kwds=dict(ls="none"),
                              scatter_kwds=dict(color=colors[p]))

            axs[p].spines["top"].set_visible(False)
            axs[p].spines["right"].set_visible(False)
            axs[p].set_title(param_name, size=18)
            axs[p].set_xlabel("Mean (in s)")
            axs[p].set_ylabel("Difference (in s)")
            axs[p].set_xticks(np.arange(0, 3.5, 0.5))
            axs[p].xaxis.set_minor_locator(plt.MultipleLocator(0.1))

        axs[0].set_ylim((-0.20, 0.20))
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=16)

        filepath = os.path.join(path, foot + '_parameter_metrics' + ".png")
        plt.savefig(filepath, format="png", bbox_inches="tight")
        plt.close()

def plot_confusion(stats: Dict, path: Optional[str] = None):
    cm = 1 / 2.54
    fig, axs = plt.subplots(2, 2, figsize=(43 * cm, 43 * cm), sharey=True)
    axs = axs.flatten()

    for i, (event, event_stats) in enumerate(stats.items()):
        confusion = np.array([
            [event_stats['true positives'], event_stats['false negatives']],
            [event_stats['false positives'], 0],
        ])

        sns.heatmap(confusion, annot=confusion, fmt='d', cmap='Blues',
                    xticklabels=['Detected Event', 'Undetected Event'],
                    yticklabels=['Event', 'No Event'],
                    ax=axs[i], cbar=False)

        axs[i].set_title(f'{event} Detection\n'
                         f'Precision: {event_stats["precision"]:.2f},'
                         f' Recall: {event_stats["recall"]:.2f}, '
                         f'F1-Score: {event_stats["f1 score"]:.2f}')

    plt.tight_layout()

    filepath = os.path.join(path, 'confusion_matrix' + '.png')
    plt.savefig(filepath, format="png", bbox_inches="tight")
    plt.close()

