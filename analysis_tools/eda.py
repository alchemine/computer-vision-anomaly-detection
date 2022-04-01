"""Exploratory Data Analysis tools

This module contains functions and classes for exploratory data analysis.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *


def _plot_on_ax(plot_fn, suptitle, ax, dir_path, figsize, show_plot):
    """Plot on a single axis if ax is not None. Otherwise, plot on a new figure.

    Parameters
    ----------
    plot_fn : function
        Function to plot.

    suptitle : str
        Title of the plot.

    ax : matplotlib.axes.Axes or list of matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.
    """
    if ax is not None:
        plot_fn(ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        with FigProcessor(fig, dir_path, show_plot, suptitle):
            plot_fn(ax)


### Missing value
def plot_missing_value(data, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot counts of missing values of each feature.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, None], 'b': [1, None, 3, 4, 5], 'c': [None, 2, 3, 4, 5]})
    >>> eda.plot_missing_value(data, dir_path='.')
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Missing value"):
        msno.matrix(data, ax=axes[0])
        ms = data.isnull().sum()
        sns.barplot(ms.index, ms, ax=axes[1])
        axes[1].bar_label(axes[1].containers[0])
        axes[1].set_xticklabels([])


### Features
def plot_features(data, bins=BINS, n_cols=N_COLS,                          dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot histogram or bar for all features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    bins : int
        Number of bins.

    n_cols : int
        Number of columns.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_features(data, dir_path='.')
    """
    n_features = len(data.columns)
    n_rows     = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Features"):
        for ax, f in zip(axes.flat, data):
            data_f_notnull = data[f].dropna()
            ax.set_title(f)
            if data_f_notnull.nunique() > bins:
                ## Numerical feature or categorical feature(many classes)
                try:
                    ax.hist(data_f_notnull, bins=bins, density=True, color='olive', alpha=0.5)
                except Exception as e:
                    print(f"[{f}]: {e}")
            else:
                ## Categorical feature(a few classes)
                cnts = data[f].value_counts(normalize=True).sort_index()  # normalize including NaN
                ax.bar(cnts.index, cnts.values, width=0.5, alpha=0.5)
                ax.set_xticks(cnts.index)
def plot_features_target(data,  target, n_cols=N_COLS,                     dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot features vs target.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.
        Dtypes of numerical features should be `number`(`numpy.float32` is recommended).
        Dtypes of categorical features should be `category`.

    target : str
        Target feature.

    dir_path : str
        Directory path to save the plot.

    n_cols : int
        Number of columns.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 1, 2], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': ['a10', 'b22', 'c11', 'a10', 'b22']})
    >>> num_features = ['a']
    >>> cat_features = data.columns.drop(num_features)
    >>> data[num_features] = data[num_features].astype(np.float32)
    >>> data[cat_features] = data[cat_features].astype('category')
    >>> eda.plot_features_target(data, 'a', dir_path='.')
    """
    num_features = data.select_dtypes('number').columns
    target_type  = 'num' if target in num_features else 'cat'
    n_rows       = int(np.ceil(len(data.columns)/n_cols))
    fig, axes    = plt.subplots(n_rows, n_cols, figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Features vs Target"):
        for ax, f in zip(axes.flat, data.columns.drop(target)):
            ax.set_title(f"{f} vs {target}")
            f_type = 'num' if f in num_features else 'cat'
            eval(f"plot_{f_type}_{target_type}_features")(data, f, target, ax=ax)
def plot_corr(corr,                                                        dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot correlation matrix.

    Parameters
    ----------
    corr : pandas.DataFrame
        Correlation matrix.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 1, 2], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [10, 20, 30, 10, 20]})
    >>> eda.plot_corr(corr, dir_path='.')
    """
    fig, ax = plt.subplots(figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Correlation matrix"):
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f", cmap='coolwarm', center=0)
def plot_num_feature(data_f, bins=BINS,                           ax=None, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot histogram of a numeric feature.

    Parameters
    ----------
    data_f : pandas.Series
        Series to be analyzed.

    bins : int
        Number of bins.

    ax : matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_num_feature(data['a'], dir_path='.')
    """
    def plot_fn(ax):
        sns.histplot(data_f, bins=bins, ax=ax, kde=True, stat='density')
        ax.set_xlabel(None)
    _plot_on_ax(plot_fn, data_f.name, ax, dir_path, figsize, show_plot)
def plot_cat_feature(data_f,                                      ax=None, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot bar of a categorical feature.

    Parameters
    ----------
    data_f : pandas.Series
        Series to be analyzed.

    ax : matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_cat_feature(data['b'], dir_path='.')
    """
    def plot_fn(ax):
        density = data_f.value_counts(normalize=True).sort_index()
        sns.barplot(density.index, density.values, ax=ax)
    _plot_on_ax(plot_fn, data_f.name, ax, dir_path, figsize, show_plot)
def plot_num_num_features(data, f1, f2, bins=BINS,                ax=None, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot histogram of two numeric features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First numerical feature.

    f2 : str
        Second numerical feature.

    bins : int
        Number of bins.

    ax : matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_num_num_features(data, 'a', 'c', dir_path='.')
    """
    def plot_fn(ax):
        sns.histplot(x=data[f1], y=data[f2], bins=bins, ax=ax)
        ax.set_xlabel(None);  ax.set_ylabel(None)
    _plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, dir_path, figsize, show_plot)
def plot_num_cat_features(data, f1, f2, n_classes=N_CLASSES_PLOT, ax=None, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot violinplot of categorical, numerical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First numerical feature.

    f2 : str
        Second categorical feature.

    n_classes : int
        Number of classes to plot.

    ax : matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_num_cat_features(data, 'a', 'b', dir_path='.')
    """
    def plot_fn(ax):
        selected_classes = data[f2].value_counts().index[:n_classes]
        idxs_selected    = data[f2][data[f2].isin(selected_classes)].index
        data_f1, data_f2 = data[f1][idxs_selected], data[f2][idxs_selected]
        sns.violinplot(x=data_f1, y=data_f2, ax=ax, orient='h', order=reversed(sorted(selected_classes)), cut=0)
        ax.set_xlabel(None);  ax.set_ylabel(None)
    _plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, dir_path, figsize, show_plot)
def plot_cat_num_features(data, f1, f2, n_classes=N_CLASSES_PLOT, ax=None, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot violinplot of categorical, numerical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First categorical feature.

    f2 : str
        Second numerical feature.

    n_classes : int
        Number of classes to plot.

    ax : matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_cat_num_features(data, 'b', 'a', dir_path='.')
    """
    def plot_fn(ax):
        selected_classes = data[f1].value_counts().index[:n_classes]
        idxs_selected    = data[f1][data[f1].isin(selected_classes)].index
        data_f1, data_f2 = data[f1][idxs_selected], data[f2][idxs_selected]
        sns.violinplot(x=data_f1, y=data_f2, ax=ax, orient='v', order=sorted(selected_classes), cut=0)
        ax.set_xlabel(None);  ax.set_ylabel(None)
    _plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, dir_path, figsize, show_plot)
def plot_cat_cat_features(data, f1, f2, n_classes=N_CLASSES_PLOT, ax=None, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot heatmap of two categorical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First categorical feature.

    f2 : str
        Second categorical feature.

    n_classes : int
        Number of classes to plot.

    ax : matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': ['a10', 'b22', 'c11', 'a10', 'b22']})
    >>> eda.plot_cat_cat_features(data, 'b', 'c', dir_path='.')
    """
    def plot_fn(ax):
        ratio = pd.crosstab(data[f2], data[f1], normalize='columns')
        ratio.sort_index(inplace=True, ascending=False)  # sort by index
        ratio = ratio[sorted(ratio)]                     # sort by column
        ratio = ratio.iloc[:n_classes, :n_classes]
        sns.heatmap(ratio, ax=ax, annot=True, fmt=".2f", cmap=sns.light_palette('firebrick', as_cmap=True), cbar=False)
        ax.set_xlabel(None);  ax.set_ylabel(None)
    _plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, dir_path, figsize, show_plot)
