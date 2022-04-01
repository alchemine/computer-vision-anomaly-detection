"""Modeling analysis tools

Modeling wrapping functions or classes are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dtreeviz.trees import dtreeviz


def get_scaling_model(model, scaler=StandardScaler()):
    """
    Creates a pipeline that applies the given scaler to the given model.

    Parameters
    ----------
    model : sklearn model
        sklearn model.

    scaler : sklearn scaler
        sklearn scaler.

    Returns
    -------
    scaled sklearn model
    """
    return make_pipeline(scaler, model)


def save_tree_visualization(fitted_model, X, y, file_path, feature_names=None, class_names=None, orientation='LR', test_sample=None):
    """
    Save a dtreeviz visualization of the given model.

    Parameters
    ----------
    fitted_model : sklearn model
        sklearn model fitted.

    X : pandas.dataframe or numpy.array
        Feature array

    y : pandas.series or numpy.array
        Target array

    file_path : string
        Path to save the dtreeviz visualization. file_path must end with '.svg'.

    feature_names : list of strings
        List of feature names.

    class_names : list of strings
        List of class names.

    orientation : string
        Orientation of the tree.
        'LR' for left to right, 'TB' for top to bottom.

    test_sample : pandas.series or numpy.array
        One sample of test data

    Examples
    --------
    >>> from analysis_tools.modeling import *
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier

    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> model = DecisionTreeClassifier(max_depth=3)
    >>> model.fit(X, y)

    >>> save_tree_visualization(model, X, y, 'iris_tree.svg', feature_names=iris.feature_names, class_names=list(iris.target_names), test_sample=X[0])
    """
    viz = dtreeviz(fitted_model, X, y, feature_names=feature_names, class_names=class_names, orientation=orientation, X=test_sample)
    assert file_path.endswith('.svg'), 'file_path must end with .svg'
    viz.save(file_path)
