"""Utility module

Commonly used utility functions or classes are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common.config import *
from analysis_tools.common.env import *


def print_guide():
    """Print machine learning project guide for the user.
    """
    print(f"┌{' MACHINE LEARNING PROJECT GUIDE ':─^50}┐")
    print(f"│{' 1. Load data':<50}│")
    print(f"│{'    1.1 Define target':<50}│")
    print(f"│{' 2. Separate training, validation, test data':<50}│")
    print(f"│{' 3. Exploratory Data Analysis(EDA)':<50}│")
    print(f"│{'    3.1 Missing value':<50}│")
    print(f"│{'    3.2 Copy data':<50}│")
    print(f"│{'    3.3 Explore features':<50}│")
    print(f"│{'    3.4 Pair plot':<50}│")
    print(f"│{' 4. Preprocessing':<50}│")
    print(f"│{'    4.1 Split X, y':<50}│")
    print(f"│{'    4.2 Imputing':<50}│")
    print(f"│{'    4.3 Detailed preprocessing(feedback loop)':<50}│")
    print(f"│{' 5. Model selection':<50}│")
    print(f"│{' 6. Model tuning':<50}│")
    print(f"│{' 7. Evaluate the model on test data':<50}│")
    print(f"└{'─' * 50}┘ \n\n")


### lambda functions
tprint  = lambda dic: print(tabulate(dic, headers='keys', tablefmt='psql'))  # print with fancy 'psql' format
ls_all  = lambda path: [path for path in glob(f"{path}/*")]
ls_dir  = lambda path: [path for path in glob(f"{path}/*") if isdir(path)]
ls_file = lambda path: [path for path in glob(f"{path}/*") if isfile(path)]
figsize = lambda x, y: (int(x*FIGSIZE_UNIT), int(y*FIGSIZE_UNIT))


@dataclass
class Timer(ContextDecorator):
    """Context manager for timing the execution of a block of code.

    Parameters
    ----------
    name : str
        Name of the timer.

    Examples
    --------
    >>> from time import sleep
    >>> from analysis_tools.common.util import Timer
    >>> with Timer('Code1'):
    ...     sleep(1)
    ...
    * Code1: 1.00s (0.02m)
    """
    name: str = ''
    def __enter__(self):
        """Start timing the execution of a block of code.
        """
        self.start_time = time()
        return self
    def __exit__(self, *exc):
        """Stop timing the execution of a block of code.

        Parameters
        ----------
        exc : tuple
            Exception information.(dummy)
        """
        elapsed_time = time() - self.start_time
        print(f"* {self.name}: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        return False


class FigProcessor(ContextDecorator):
    """Context manager for processing figure.

    Plot the figure and save it to the specified path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to be processed.

    dir_path : str
        Directory path to save the figure.

    show_plot : bool
        Whether to show the figure.

    suptitle : str
        Super title of the figure.

    suptitle_options : dict
        Options for super title.

    tight_layout : bool
        Whether to use tight layout.

    Examples
    --------
    >>> from analysis_tools.common.util import FigProcessor
    >>> fig, ax = plt.subplots()
    >>> with FigProcessor(fig, suptitle="Feature distribution"):
    ...     ax.plot(...)
    """
    def __init__(self, fig, dir_path, show_plot=SHOW_PLOT, suptitle='', suptitle_options={}, tight_layout=True):
        self.fig              = fig
        self.dir_path         = dir_path
        self.show_plot        = show_plot
        self.suptitle         = suptitle
        self.suptitle_options = suptitle_options
        self.tight_layout     = tight_layout
    def __enter__(self):
        pass
    def __exit__(self, *exc):
        """Save and plot the figure.

        Parameters
        ----------
        exc : tuple
            Exception information.(dummy)
        """
        if self.tight_layout:
            self.fig.suptitle(self.suptitle, **self.suptitle_options)
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        if self.dir_path:
            self.fig.savefig(join(self.dir_path, f"{self.suptitle}.png"))
        if self.show_plot:
            plt.show()
        plt.close(self.fig)
