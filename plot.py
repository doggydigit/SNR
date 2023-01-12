import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.colors import LogNorm
from pathlib import Path
from scipy import stats

import scan
import utils.paths
from utils import ids, keys, sql, databases
from scan import NR_TRIALS
from utils.paths import RESULT_DIR

PLOT_DIR = Path("plots")
SVG = "svg"


def get_outcome(simulation_type: ids.SimulationType, noise_std: float, nr_noise: int, learning_rate: float,
                nr_useful: int, optimizer_type: ids.OptimizerID) -> np.ndarray:
    get_cols = [f"{keys.LOSS}_{n}" for n in range(NR_TRIALS)]
    db_path = RESULT_DIR / "scan.db"
    table = simulation_type

    where_cols: list[str] = [keys.LEARNING_RATE, keys.NOISE_STD, keys.NUMBER_NOISE, keys.NUMBER_SIGNAL, keys.OPTIMIZER]
    values = (learning_rate, noise_std, nr_noise, nr_useful, optimizer_type)
    where_values = {where_col: value for where_col, value in zip(where_cols, values)}
    print(where_values)
    raise None

    rows = sql.select(db_path=db_path, table=table, get_cols=get_cols, where_values=where_values)

    if len(rows) == 0:
        raise FileNotFoundError
    else:
        return np.array([row[0] for row in rows])


def landscape(simulation_type: ids.SimulationType, optimizer: ids.OptimizerID, nr_signal: int, best_lr: bool = False,
              save: bool = False) -> None:

    # Plotting preparations
    plot_dir = PLOT_DIR
    Path(plot_dir / SVG).mkdir(parents=True, exist_ok=True)
    xlab = "Noise Amplitude"  # TODO: Quantify to SNR
    ylab = "Number Noise"
    if simulation_type in ids.LOSS_SIMULATIONS:
        if simulation_type == ids.REGRESSION:
            title = "Loss (Regression)"
        elif simulation_type == ids.TD_SALIENCE_NAIVE:
            title = "Loss (Naive Salience Prediction)"
        else:
            raise NotImplementedError(simulation_type)
        cmap_color = "PiYG_r"
        max_v = 1.
    elif simulation_type in ids.PERF_SIMULATIONS:
        if simulation_type == ids.GONOGO:
            title = "Performance (GO/NOGO)"
        else:
            raise NotImplementedError(simulation_type)
        cmap_color = "PiYG"
        max_v = 0.5
    else:
        raise ValueError(simulation_type)

    get_cols: list[keys.Key] = [keys.NOISE_STD, keys.SIGNAL_AMPLITUDE, keys.NUMBER_NOISE]
    if best_lr:
        if simulation_type in ids.PRIMARY_SIMULATIONS:
            get_cols += [keys.LEARNING_RATE]
            max_v = 0.5
        else:
            raise NotImplementedError(simulation_type)
        cmap_color = "winter"
        v = np.array(list(reversed([0.5 ** p for p in np.linspace(1., 9., 101, endpoint=True)])))
        v_tick = list(reversed([0.5 ** p for p in np.linspace(1., 9., 5, endpoint=True)]))
    else:
        get_cols += databases.get_outcome_cols(simulation_type=simulation_type, table=databases.BEST_LR_TABLE)
        v = np.linspace(0., max_v, 101, endpoint=True)
    locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=5)
    minx, maxx, miny, maxy = 1., 0., 1., 0.

    # Iterate over all desired datasets and models
    table = databases.BEST_LR_TABLE
    where_values = {}
    db_path = databases.get_db_path(folder=utils.paths.RESULT_DIR, simulation_type=simulation_type,
                                    optimizer=optimizer)
    where_values[keys.NUMBER_SIGNAL] = 1  # TODO: Adapt for each scan type
    fig, axs = plt.subplots()
    rows = sql.select(db_path=db_path, table=table, get_cols=get_cols, where_values=where_values)
    snrs = [(row[1]/row[0])**2 for row in rows]
    nr_noises = [row[2] for row in rows]
    # outcomes = np.array([min(float(row[3])/8, max_v-0.000001) if isinstance(row[2], float) else max_v + 1. for row in rows])
    outcomes = np.array([min(float(row[3]), max_v-0.000001) for row in rows])

    # Plot data
    if best_lr:
        cntr = axs.tricontourf(snrs, nr_noises, outcomes, v, cmap=cmap_color, extend="neither", norm=LogNorm())
    else:
        cntr = axs.tricontourf(snrs, nr_noises, outcomes, v, cmap=cmap_color, extend="neither")

    axs.plot(snrs, nr_noises, 'ko', ms=1)

    # Get the min and max range of the learning rates across all architectures
    current_m = min(snrs)
    minx = minx if current_m > minx else current_m
    current_m = max(snrs)
    maxx = maxx if current_m < maxx else current_m
    current_m = min(nr_noises)
    miny = miny if current_m > miny else current_m
    current_m = max(nr_noises)
    maxy = maxy if current_m < maxy else current_m
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.xaxis.set_minor_locator(locmin)
    axs.xaxis.set_minor_formatter(NullFormatter())
    axs.yaxis.set_minor_locator(locmin)
    axs.yaxis.set_minor_formatter(NullFormatter())
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(cntr, cax=cbar_ax)
    if best_lr:
        v_tick = list(reversed([0.5 ** p for p in np.linspace(1., 9., 5, endpoint=True)]))
        cbar.set_ticks(v_tick)
        cbar.set_ticklabels([f"{vi:.1E}" for vi in v_tick])
    else:
        if simulation_type in ids.LOSS_SIMULATIONS:
            cbar.set_ticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
            # cbar.set_ticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
        else:
            cbar.set_ticks([0., 0.1, 0.2, 0.3, 0.4, 0.5])

    # Plotting Specifics
    axs.set_xlabel(xlab)
    axs.set_ylabel(ylab)
    fig.set_size_inches(6, 6)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.13, right=0.90, wspace=0.15, hspace=0.26)
    plt.suptitle(title)

    if save:
        if best_lr:
            plt.savefig(plot_dir / SVG / f"scan_{simulation_type}_{optimizer}_{nr_signal}_best_lr.svg")
        else:
            plt.savefig(plot_dir / SVG / f"scan_{simulation_type}_{optimizer}_{nr_signal}.svg")
            plt.savefig(plot_dir / f"scan_{simulation_type}_{optimizer}_{nr_signal}.png")
        plt.close()
    else:
        plt.show()


def plot_scan_type(scan_type)

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    blr = True
    landscape(simulation_type=ids.REGRESSION, save=True, best_lr=blr)
    landscape(simulation_type=ids.GONOGO, save=True, best_lr=blr)
    landscape(simulation_type=ids.TD_SALIENCE_NAIVE, save=True, best_lr=blr)
