"""Modeling analysis tools

Modeling wrapping functions or classes are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    from dtreeviz.trees import dtreeviz

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

def get_callbacks(patience=10, plot_path=None, warmup_epoch=None, init_lr=2e-3, epochs=100, min_lr=1e-3, power=1):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard

    class LearningPlot(Callback):
        """
        Plot learning curve for every epoch
        """

        def __init__(self, plot_path, figsize=(15, 8)):
            self.plot_path = plot_path
            self.figsize = figsize
            self.metrics = pd.DataFrame()
            self.best_epoch = -1
            mkdir(dirname(self.plot_path))

        def on_epoch_end(self, epoch, logs={}):
            self.metrics = self.metrics.append(logs, ignore_index=True)
            if 'val_loss' in self.metrics:
                self.best_epoch = np.argmin(self.metrics['val_loss'])
            self._save_fig(epoch)

        def _save_fig(self, epoch):
            fig, ax_loss = plt.subplots(figsize=self.figsize)
            ax_metric = ax_loss.twinx()

            lns = []
            for col in self.metrics:
                ax = ax_loss if 'loss' in col else ax_metric
                lns += ax.plot(self.metrics[col], color=('r' if col.startswith('val') else 'b'),
                               linestyle=('-' if 'loss' in col else '--'), label=col)

            if self.best_epoch != -1:
                ln = ax_loss.axvline(self.best_epoch, color='k', ls='-.', lw=2, label=f'best_epoch: {self.best_epoch}')
                lns += [ln]

            for ax in (ax_loss, ax_metric):
                yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 7)
                ax.set_yticks(yticks)

            ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_loss.set_xticklabels([str(int(x) + 1) for x in ax_loss.get_xticks()])

            ax_loss.set_xlabel('epoch')
            ax_loss.set_ylabel('loss');
            ax_metric.set_ylabel('metric')
            # ax_loss.set_yscale('log')

            ax_loss.legend(lns, [ln.get_label() for ln in lns], loc='upper left', bbox_to_anchor=(1.1, 1))
            ax_loss.set_title(split(self.plot_path)[1], fontsize='x-large', fontweight='bold')
            fig.tight_layout()
            fig.savefig(f'{self.plot_path}.png')
            plt.close(fig)

    class LRScheduler(Callback):
        def __init__(self, init_lr, epochs, warmup_epoch, min_lr, power):
            self.init_lr = init_lr
            self.epochs = epochs
            self.warmup_epoch = warmup_epoch
            self.min_lr = min_lr
            self.power = power
            self.decay_fn = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=init_lr,
                decay_steps=epochs - warmup_epoch,
                end_learning_rate=min_lr,
                power=power
            )
            self.lrs = []

        def on_epoch_begin(self, epoch, logs=None):
            global_epoch = tf.cast(epoch + 1, tf.float64)
            warmup_epoch_float = tf.cast(self.warmup_epoch, tf.float64)

            lr = tf.cond(
                global_epoch < warmup_epoch_float,
                lambda: self.init_lr * (global_epoch / warmup_epoch_float),
                lambda: self.decay_fn(global_epoch - warmup_epoch_float),
            )
            tf.print('learning rate: ', lr)
            keras.backend.set_value(self.model.optimizer.lr, lr)
            self.lrs.append(lr)

    callbacks = []
    callbacks.append(EarlyStopping(patience=patience, verbose=1))
    callbacks.append(TensorBoard(join(PATH.root, 'tensorboard')))
    if plot_path:
        callbacks.append(LearningPlot(plot_path))
    if warmup_epoch:
        callbacks.append(LRScheduler(init_lr, epochs, warmup_epoch, min_lr, power))
    return callbacks
