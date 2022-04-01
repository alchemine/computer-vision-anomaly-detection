"""Metric analysis tools

Performance evaluation metrics are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


def confusion_matrix_analysis(y_true, y_pred, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot confusion matrix

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    figsize : tuple
        Figure size.

    dir_path : str
        Path to save the figure.

    show_plot : bool
        Whether to show the figure.

    Returns
    -------
    collections of metrics : dictionary
        Confusion matrix, accuracy, precision, recall, f1 score

    Examples
    --------
    >>> from analysis_tools.metrics import confusion_matrix_analysis
    >>> y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 0, 1, 1, 1, 1, 1]
    >>> confusion_matrix_analysis(y_true, y_pred, dir_path='.')
    """
    normalized_C = confusion_matrix(y_true, y_pred, normalize='true')
    assert all(normalized_C.sum(axis=1) == 1), "Confusion matrix is not normalized"

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Confusion matrix"):
        sns.heatmap(normalized_C, annot=False, fmt='.2%', cmap='gray', ax=axes[0, 0])
        sns.heatmap(normalized_C, annot=True, fmt='.2%', cmap='gray', ax=axes[0, 1])

        normalized_C_off_diagonal = copy(normalized_C)
        np.fill_diagonal(normalized_C_off_diagonal, 0)  # off-diagonal
        sns.heatmap(normalized_C_off_diagonal, annot=False, fmt='.2%', cmap='gray', ax=axes[1, 0])
        sns.heatmap(normalized_C_off_diagonal, annot=True, fmt='.2%', cmap='gray', ax=axes[1, 1])
        for ax in axes.flat:
            ax.xaxis.tick_top()
    return dict(
        confusion_matrix=normalized_C,
        accuracy=accuracy_score(y_true, y_pred), precision=precision_score(y_true, y_pred), recall=recall_score(y_true, y_pred), f1_score=f1_score(y_true, y_pred),
    )
def curve_analysis(y_true, y_score,           dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot Precision-Recall and ROC curves

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,)
        Estimated probabilities or output of a decision function.

    dir_path : str
        Path to save the figure.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the figure.

    Examples
    --------
    >>> from analysis_tools.metrics import curve_analysis
    >>> y_true  = [0, 0, 0, 0, 1, 1, 1, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8, 0.85, 0.8, 0.9, 0.95]
    >>> curve_analysis(y_true, y_score, dir_path='.')
    """
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_score)
    fpr, tpr, thresholds_roc           = roc_curve(y_true, y_score)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Precision-Recall & ROC curves"):
        # Thresholds-PR
        axes[0].show_plot(thresholds_pr, precisions[:-1], 'b--', label='Precision')
        axes[0].show_plot(thresholds_pr, recalls[:-1], 'g-', label='Recall')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Precision/Recall')
        axes[0].set_ylim([0, 1])
        axes[0].legend()

        # Precision-Recall
        axes[1].show_plot(recalls, precisions, label=f"PR-AUC: {average_precision_score(y_true, y_score):.3f}")
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].legend()

        # ROC
        axes[2].show_plot(fpr, tpr, linewidth=2, label=f"ROC-AUC: {roc_auc_score(y_true, y_score):.3f}")
        axes[2].show_plot([0, 1], [0, 1], 'k--')
        axes[2].set_xlabel('FPR(=FP/RealNegative)')
        axes[2].set_ylabel('TPR(Recall)')
        axes[2].set_xlim([0, 1])
        axes[2].set_ylim([0, 1])
        axes[2].legend()

def get_feature_importance(data, target, bins=BINS, problem='classification', dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Get feature importance using RandomForest model.

    The metrics are mean decrease in impurity, mean accuracy decrease, mean rank

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    target : str
        Target feature.

    bins : int
        Number of bins.

    problem : str
        Problem type.(`classification` or `regression`)

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Returns
    -------
    pandas.DataFrame
        Feature importances.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 1, 2], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [10, 20, 30, 10, 20]})
    >>> num_features = ['c']
    >>> cat_features = data.columns.drop(num_features)
    >>> data[num_features] = data[num_features].astype(np.float32)
    >>> data[cat_features] = data[cat_features].astype('category')
    >>> eda.get_feature_importance(data, 'a', dir_path='.')
    """
    ## 1. Split data into X, y
    data               = data.dropna()
    cat_features       = data.select_dtypes('category').columns
    data[cat_features] = data[cat_features].apply(OrdinalEncoder().fit_transform)
    X, y = data.drop(columns=target), data[target]

    ## 2. Model
    model = RandomForestClassifier(n_jobs=-1) if problem == 'classification' else RandomForestRegressor(n_jobs=-1)
    model.fit(X, y)

    ## 3. Get feature importance
    MDI_importance  = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    perm_importance = pd.Series(permutation_importance(model, X, y).importances_mean, index=X.columns).sort_values(ascending=False)

    ## 4. Mean importance
    fi1     = pd.Series(range(len(MDI_importance)), index=MDI_importance.index, name='MDI')
    fi2     = pd.Series(range(len(perm_importance)), index=perm_importance.index, name='Permutation')
    mean_fi = pd.Series(((fi1 + fi2)/2).sort_values(), name='Mean')

    ## 5. Plot
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Feature importance"):
        for ax, data, ylabel, title in zip(axes,
                                          [MDI_importance.head(bins), perm_importance.head(bins), mean_fi.head(bins)],
                                          ["Mean decrease in impurity", "Mean accuracy decrease", "Mean rank"],
                                          ["Feature importance using MDI", "Feature importance using permutation on full model", "Feature importance using MDI, permutation on full model"]):
            sns.barplot(data.index, data, ax=ax)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=30)

    return pd.concat([MDI_importance, perm_importance, mean_fi], axis='columns')

def plot_learning_curve(model, X_train, y_train, X_val, y_val, n_subsets_step=LEARNING_CURVE_N_SUBSETS_STEP, problem='classification', dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot learning curve

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Model to train.

    X_train : array-like of shape (n_samples, n_features)
        Training data.

    y_train : array-like of shape (n_samples,)
        Training target values.

    X_val : array-like of shape (n_samples, n_features)
        Validation data.

    y_val : array-like of shape (n_samples,)
        Validation target values.

    n_subsets_step : int
        Step size for subsets.

    problem : str
        Problem type.(`classification` or `regression`)

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> from analysis_tools.metrics import plot_learning_curve
    >>> from sklearn.linear_model import LogisticRegression
    >>> X_train = [[0, 0], [1, 1]]
    >>> y_train = [0, 1]
    >>> X_val = [[0, 0], [1, 1]]
    >>> y_val = [0, 1]
    >>> model = LogisticRegression()
    >>> plot_learning_curve(model, X_train, y_train, X_val, y_val)
    """
    if problem == 'classification':
        error_fn_names = ['F1 score', 'Precision', 'Recall', 'Accuracy']
        error_fns      = [precision_score, recall_score, f1_score, accuracy_score]
        fig, axes = plt.subplots(4, 1, figsize=figsize)
    else:
        error_fn_names = ['MSE', 'R-squared']
        error_fns      = [mean_squared_error, r2_score]
        fig, axes      = plt.subplots(2, 1, figsize=figsize)

    with FigProcessor(fig, dir_path, show_plot, "Learning curve"):
        for ax, error_fn_name, error_fn in zip(axes, error_fn_names, error_fns):
            train_sub_errors, val_errors = pd.Series([], name='Training error'), pd.Series([], name='Validation error')
            for n_subsets in trange(1, len(X_train), n_subsets_step):
                try:
                    X_train_sub, y_train_sub = X_train[:n_subsets], y_train[:n_subsets]
                    model.fit(X_train_sub, y_train_sub)
                    y_train_sub_pred = model.predict(X_train_sub)
                    y_val_pred       = model.predict(X_val)
                    train_sub_errors.loc[n_subsets] = error_fn(y_train_sub, y_train_sub_pred)
                    val_errors.loc[n_subsets]       = error_fn(y_val, y_val_pred)
                except ValueError as e:
                    print(e)
                    train_sub_errors.loc[n_subsets] = np.nan
                    val_errors.loc[n_subsets]       = np.nan
            ax.plot(train_sub_errors, 'r-+', linewidth=2)
            ax.plot(val_errors, 'b-', linewidth=3)
            ax.legend()
            ax.grid()
            ax.set_ylabel(error_fn_name)
            if ax == axes[-1]:
                ax.set_xlabel('Number of training samples')
            else:
                ax.set_xticklabels([])

def compare_models(models, X_train, y_train, X_val=None, y_val=None):
    """Compare models with `model.score()`

    Parameters
    ----------
    models : list of sklearn.base.BaseEstimator
        Models to compare.

    X_train : array-like of shape (n_samples, n_features)
        Training data.

    y_train : array-like of shape (n_samples,)
        Training target values.

    X_val : array-like of shape (n_samples, n_features)
        Validation data.

    y_val : array-like of shape (n_samples,)
        Validation target values.

    Returns
    -------
    list of tuple of (score, model)

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.svm import SVR
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> from analysis_tools.metrics import compare_models
    >>>
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    >>> scaler = StandardScaler()
    >>> X_train = scaler.fit_transform(X_train)
    >>> X_val  = scaler.transform(X_val)

    >>> base_models = [
    ...    LinearRegression(n_jobs=-1),
    ...    SVR(),
    ...    RandomForestRegressor(n_jobs=-1),
    ...    KNeighborsRegressor(n_jobs=-1)
    ... ]
    >>> ensemble_models = [
    ...    VotingRegressor([(model.__class__.__name__, model) for model in base_models], n_jobs=-1)
    ... ]

    >>> models = base_models + ensemble_models
    >>> results = compare_models(models, X_train, y_train, X_val, y_val)
    100%|█████████████████████████████████████████████| 5/5 [00:00<00:00,  5.11it/s]
    - Scores
    0.658 (train) / 0.467 (val) : VotingRegressor
    0.528 (train) / 0.453 (val) : LinearRegression
    0.921 (train) / 0.443 (val) : RandomForestRegressor
    0.580 (train) / 0.425 (val) : KNeighborsRegressor
    0.167 (train) / 0.182 (val) : SVR
    """
    if X_val is None and y_val is None:
        try:  # classification
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train)
        except:  # regression
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

    results = []
    for model in tqdm(models):
        model.fit(X_train, y_train)
        score_train = model.score(X_train, y_train)
        score_val   = model.score(X_val, y_val)
        results.append((score_train, score_val, model))
    results = sorted(results, key=lambda result: result[1], reverse=True)  # sort by score_val

    print("- Scores")  # accuracy(regression), r-square(classification)
    for score_train, score_val, model in results:
        print(f"{score_train:.3f} (train) / {score_val:.3f} (val) : {model.__class__.__name__}")
    return results
