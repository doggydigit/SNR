import warnings
import sqlite3
from typing_extensions import NotRequired, TypedDict
from utils import databases, ids, keys, paths, sql


DEFAULT_LR = 0.1


class Parameters(TypedDict):
    simulation_type: ids.SimulationType
    optimizer: ids.OptimizerID
    nr_useful: int
    noise_std: float
    nr_noise: int
    learning_rate: float
    learning_rate_td: NotRequired[float]
    nr_noise_td: NotRequired[int]
    nr_useful_td: NotRequired[int]


def get_best_lr(simulation_type: ids.SimulationType, optimizer: ids.OptimizerID, nr_useful: int, noise_std: float,
                nr_noise: int, learning_rate: float = None, nr_noise_td: int = None,
                nr_useful_td: int = None, verbose=True) -> (bool, float):

    db_path = databases.get_db_path(folder=paths.RESULT_DIR, simulation_type=simulation_type, optimizer=optimizer)
    table = databases.BEST_LR_TABLE

    # The other necessary specifications to get the optimal params (architecture specific, task specific, dataset)
    if simulation_type in ids.SECONDARY_SIMULATIONS:
        assert learning_rate is not None
        assert nr_noise_td is not None
        assert nr_useful_td is not None
        get_cols = [keys.LEARNING_RATE_TD]
        where_values = {keys.NOISE_STD: noise_std, keys.NUMBER_NOISE: nr_noise, keys.NUMBER_SIGNAL: nr_useful,
                        keys.LEARNING_RATE: learning_rate, keys.NUMBER_NOISE_TD: nr_noise_td,
                        keys.NUMBER_SIGNAL_TD: nr_useful_td}
    else:
        get_cols = [keys.LEARNING_RATE]
        where_values = {keys.NOISE_STD: noise_std, keys.NUMBER_NOISE: nr_noise, keys.NUMBER_SIGNAL: nr_useful}

    # Retrieve the best learning rates from the database
    try:
        rows = sql.select(db_path=db_path, table=table, get_cols=get_cols, where_values=where_values)
    except sqlite3.OperationalError:
        if verbose:
            warnings.warn(f"Database '{db_path}' has no table {table} for optimal lr")
        return False, DEFAULT_LR

    # Print some warnings if no or multiple sets of learning rates fit the search
    if len(rows) == 0:
        if verbose:
            warnings.warn(f"Simulation {simulation_type} with optimizer {optimizer} and parameters {where_values} has "
                          f"no optimal learning rate")
        return False, DEFAULT_LR

    elif len(rows) >= 2:
        if verbose:
            warnings.warn(f"Simulation {simulation_type} with optimizer {optimizer} and parameters {where_values} has "
                          f"multiple entries of optimal training parameters. Randomly selected one.")

    return True, rows[0][0]
