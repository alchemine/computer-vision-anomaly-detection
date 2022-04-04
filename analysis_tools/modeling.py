"""Modeling analysis tools

Modeling wrapping functions or classes are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dtreeviz.trees import dtreeviz
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard
from matplotlib.ticker import MaxNLocator


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


class LearningPlot(Callback):
    """
    Plot learning curve for every epoch
    """
    def __init__(self, plot_path, figsize=(15, 8)):
        self.plot_path  = plot_path
        self.figsize    = figsize
        self.metrics    = pd.DataFrame()
        self.best_epoch = -1
        mkdir(dirname(self.plot_path))
    def on_epoch_end(self, epoch, logs={}):
        self.metrics = self.metrics.append(logs, ignore_index=True)
        if 'val_loss' in self.metrics:
            self.best_epoch = np.argmin(self.metrics['val_loss'])
        self._save_fig(epoch)
    def _save_fig(self, epoch):
        fig, ax_loss = plt.subplots(figsize=self.figsize)
        ax_metric    = ax_loss.twinx()

        lns = []
        for col in self.metrics:
            ax = ax_loss if 'loss' in col else ax_metric
            lns += ax.plot(self.metrics[col], color=('r' if col.startswith('val') else 'b'), linestyle=('-' if 'loss' in col else '--'), label=col)

        if self.best_epoch != -1:
            ln = ax_loss.axvline(self.best_epoch, color='k', ls='-.', lw=2, label=f'best_epoch: {self.best_epoch}')
            lns += [ln]

        for ax in (ax_loss, ax_metric):
            yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 7)
            ax.set_yticks(yticks)

        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_loss.set_xticklabels([str(int(x)+1) for x in ax_loss.get_xticks()])

        ax_loss.set_xlabel('epoch')
        ax_loss.set_ylabel('loss');  ax_metric.set_ylabel('metric')
        # ax_loss.set_yscale('log')

        ax_loss.legend(lns, [ln.get_label() for ln in lns], loc='upper left', bbox_to_anchor=(1.1, 1))
        ax_loss.set_title(split(self.plot_path)[1], fontsize='x-large', fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{self.plot_path}.png')
        plt.close(fig)


def get_callbacks(patience, plot_path):
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    tensorboard    = TensorBoard(join(PATH.root, 'tensorboard'))
    learning_plot  = LearningPlot(plot_path)
    return early_stopping, tensorboard, learning_plot
