import os
import sqlite3
import random
import torch
import numpy as np
from pathlib import Path
from typing import Literal
from utils import ids, keys, sql, databases, parameters
from utils.databases import MAV_N
from utils.params import NR_TRIALS
from utils import paths

NR_SEEDS_SCAN_REGRESSION = 8
NR_SEEDS_SCAN = 32
LEARNING_RATES = [0.5 ** p for p in range(1, 10)]
ScanType = Literal['vary_noise', 'vary_signal_no_noise', 'vary_signal_high_noise', 'vary_noise_and_amplitude']
ST_VARY_NOISE: ScanType = 'vary_noise'
ST_VARY_SIGNAL_NO_NOISE: ScanType = 'vary_signal_no_noise'
ST_VARY_SIGNAL_HIGH_NOISE: ScanType = 'vary_signal_high_noise'
ST_VARY_NOISE_AND_AMPLITUDE: ScanType = 'vary_noise_and_amplitude'


def simulate_regression(noise_std: float, signal_amplitude: float, tot_nr_neurons: int, useful_indices: torch.Tensor,
                        regressor: torch.nn.Module, optimizer: torch.optim.Optimizer) -> np.ndarray:
    losses = np.array([-666.666] * NR_TRIALS)
    for trial in range(NR_TRIALS):
        sensory_representation = torch.normal(mean=0., std=noise_std, size=[tot_nr_neurons])
        if sensory_representation[0] > 0.:
            value = signal_amplitude
        else:
            value = -signal_amplitude
        sensory_representation.index_fill_(dim=0, index=useful_indices, value=value)
        prediction = torch.tanh(regressor.forward(sensory_representation))
        loss = 0.5 * (value - prediction) ** 2
        losses[trial] = loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


def simulate_gonogo(noise_std: float, signal_amplitude: float, tot_nr_neurons: int, signal_indices: torch.Tensor,
                    policy: torch.nn.Module, optimizer: torch.optim.Optimizer, av_abs: bool = False) -> np.ndarray:
    rewards = np.array([-666.666] * NR_TRIALS)
    n_useful = len(signal_indices)
    go_txt_idx = signal_indices[:int(n_useful / 2)]
    nogo_txt_idx = signal_indices[int(n_useful / 2):]
    av_rew = 0.

    for trial in range(NR_TRIALS):
        sensory_representation = torch.normal(mean=0., std=noise_std, size=[tot_nr_neurons])
        go = sensory_representation[0] > 0.
        if go:
            sensory_representation.index_fill_(dim=0, index=go_txt_idx, value=signal_amplitude)
            sensory_representation.index_fill_(dim=0, index=nogo_txt_idx, value=0.)
        else:
            sensory_representation.index_fill_(dim=0, index=go_txt_idx, value=0.)
            sensory_representation.index_fill_(dim=0, index=nogo_txt_idx, value=signal_amplitude)
        lick_prob = torch.sigmoid(policy.forward(sensory_representation))
        l_prob = lick_prob.detach().numpy()[0]
        lick = bool(np.random.choice([0, 1], p=[1 - l_prob, l_prob]))
        if lick:
            if go:
                reward = 1.
            else:
                reward = -0.5
            loss = -(reward - av_rew) * torch.log(lick_prob)
        else:
            reward = 0.
            loss = av_rew * torch.log(1 - lick_prob)
        rewards[trial] = reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if av_abs:
            av_rew = 0.9 * av_rew + abs(reward) * 0.1
        else:
            av_rew = 0.9 * av_rew + reward * 0.1

    return rewards


def simulate_td_naive(noise_std: float, signal_amplitude: float, tot_nr_neurons: int, useful_indices: torch.Tensor,
                      estimator: torch.nn.Module, optimizer: torch.optim.Optimizer, salience: bool = True
                      ) -> np.ndarray:
    losses = np.array([-666.666] * NR_TRIALS)
    n_useful = len(useful_indices)
    go_txt_idx = useful_indices[:int(n_useful / 2)]
    nogo_txt_idx = useful_indices[int(n_useful / 2):]

    for trial in range(NR_TRIALS):
        sensory_representation = torch.normal(mean=0., std=noise_std, size=[tot_nr_neurons])
        go = sensory_representation[0] > 0.
        if go:
            sensory_representation.index_fill_(dim=0, index=go_txt_idx, value=signal_amplitude)
            sensory_representation.index_fill_(dim=0, index=nogo_txt_idx, value=0.)
        else:
            sensory_representation.index_fill_(dim=0, index=go_txt_idx, value=0.)
            sensory_representation.index_fill_(dim=0, index=nogo_txt_idx, value=signal_amplitude)

        estimate = estimator.forward(sensory_representation)
        lick = bool(np.random.choice([0, 1]))
        if lick:
            if go:
                reward = 1.
            else:
                reward = -1.
        else:
            reward = 0.
        if salience:
            outcome = abs(reward)
        else:
            outcome = reward
        loss = 0.5 * (estimate - outcome) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[trial] = loss.detach().item()

    return losses


def simulate_td(noise_std: float, signal_amplitude: float, tot_nr_neurons: int, useful_indices: torch.Tensor,
                estimator: torch.nn.Module, optimizer: torch.optim.Optimizer, indices_td: torch.Tensor,
                estimator_td: torch.nn.Module, optimizer_td: torch.optim.Optimizer, salience: bool = True
                ) -> np.ndarray:
    losses = np.array([-666.666] * NR_TRIALS)
    n_useful = len(useful_indices)
    go_txt_idx = useful_indices[:int(n_useful / 2)]
    nogo_txt_idx = useful_indices[int(n_useful / 2):]
    av_rew = 0.

    for trial in range(NR_TRIALS):
        sensory_representation = torch.normal(mean=0., std=noise_std, size=[tot_nr_neurons])
        go = sensory_representation[0] > 0.
        if go:
            sensory_representation.index_fill_(dim=0, index=go_txt_idx, value=signal_amplitude)
            sensory_representation.index_fill_(dim=0, index=nogo_txt_idx, value=0.)
        else:
            sensory_representation.index_fill_(dim=0, index=go_txt_idx, value=0.)
            sensory_representation.index_fill_(dim=0, index=nogo_txt_idx, value=signal_amplitude)

        lick_prob = torch.sigmoid(estimator.forward(sensory_representation))
        l_prob = lick_prob.detach().numpy()[0]
        lick = bool(np.random.choice([0, 1], p=[1 - l_prob, l_prob]))
        if lick:
            if go:
                reward = 1.
            else:
                reward = -0.5
            loss = -(reward - av_rew) * torch.log(lick_prob)
        else:
            reward = 0.
            loss = av_rew * torch.log(1 - lick_prob)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        av_rew = 0.9 * av_rew + reward * 0.1

        if salience:
            outcome = abs(reward)
        else:
            outcome = reward
        estimate_td = estimator_td.forward(sensory_representation[indices_td])
        loss_td = 0.5 * (estimate_td - outcome) ** 2
        losses[trial] = loss_td.detach().item()
        optimizer_td.zero_grad()
        loss_td.backward()
        optimizer_td.step()
        # print(go.item(), reward, f'{losses[trial]:0.2f}', estimator_td.weight.data)
    return losses


def simulate(simulation_type: ids.SimulationType, optimizer_type: ids.OptimizerID, learning_rate: float, seed: int,
             noise_std: float, signal_amplitude: float, nr_noise: int, nr_signal: int, nr_noise_td: int = None,
             nr_signal_td: int = None) -> np.ndarray:

    # Set pseudorandom seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sensory representation initialization
    useful_indices = torch.tensor(list(range(nr_signal)))
    tot_nr_neurons = nr_noise + nr_signal

    # Init policy network and optimizer
    network = torch.nn.Linear(tot_nr_neurons, 1)
    if simulation_type == ids.REGRESSION:
        opt_parameters = network.parameters()
    else:
        opt_parameters = [network.weight]
        network.bias.data.fill_(0.)

    # In case of secondary simulations (two separate estimators) use the optimal lr for the policy
    if simulation_type in ids.SECONDARY_SIMULATIONS:
        found, learning_rate_opt = parameters.get_best_lr(simulation_type=ids.GONOGO, optimizer=optimizer_type,
                                                          nr_useful=nr_signal, noise_std=noise_std, nr_noise=nr_noise)
        if not found:
            raise FileNotFoundError("need to find the optimal lr for gonogo")
    else:
        learning_rate_opt = learning_rate

    # Optimizer encompassing learning rates and parameters to train
    match optimizer_type:
        case ids.SGD:
            optimizer = torch.optim.SGD(opt_parameters, learning_rate_opt)
        case ids.ADAM:
            optimizer = torch.optim.Adam(opt_parameters, learning_rate_opt)
        case _:
            raise ValueError(optimizer_type)

    # Simulate online training
    match simulation_type:
        case ids.REGRESSION:
            outcomes = simulate_regression(noise_std=noise_std, signal_amplitude=signal_amplitude,
                                           tot_nr_neurons=tot_nr_neurons, useful_indices=useful_indices,
                                           regressor=network, optimizer=optimizer)
        case ids.GONOGO | ids.GONOGO_AVABS:
            av_abs = simulation_type == ids.GONOGO_AVABS
            outcomes = simulate_gonogo(noise_std=noise_std, signal_amplitude=signal_amplitude,
                                       tot_nr_neurons=tot_nr_neurons, signal_indices=useful_indices,
                                       policy=network, optimizer=optimizer, av_abs=av_abs)
        case ids.TD_SALIENCE_NAIVE:
            salience = simulation_type == ids.TD_SALIENCE_NAIVE
            outcomes = simulate_td_naive(noise_std=noise_std, signal_amplitude=signal_amplitude,
                                         tot_nr_neurons=tot_nr_neurons, useful_indices=useful_indices,
                                         estimator=network,
                                         optimizer=optimizer, salience=salience)

        case ids.TD_SALIENCE:
            indices_td = torch.tensor(list(range(nr_signal_td)) + list(range(nr_signal, nr_signal + nr_noise_td)))
            network_td = torch.nn.Linear(nr_signal_td + nr_noise_td, 1)
            network_td.bias.data.fill_(0.)
            parameters_td = [network_td.weight]
            match optimizer_type:
                case ids.SGD:
                    optimizer_td = torch.optim.SGD(parameters_td, learning_rate)
                case ids.ADAM:
                    optimizer_td = torch.optim.Adam(parameters_td, learning_rate)
                case _:
                    raise ValueError(optimizer_type)
            salience = simulation_type == ids.TD_SALIENCE
            outcomes = simulate_td(noise_std=noise_std, signal_amplitude=signal_amplitude,
                                   tot_nr_neurons=tot_nr_neurons,
                                   useful_indices=useful_indices, estimator=network, optimizer=optimizer,
                                   indices_td=indices_td, estimator_td=network_td, optimizer_td=optimizer_td,
                                   salience=salience)
        case _:
            raise NotImplementedError(simulation_type)

    return outcomes


def get_scan_params(scan_type: ScanType, sim_type: ids.SimulationType) -> tuple[list[float], list[float], list[int], list[int]]:
    if scan_type == ST_VARY_NOISE:
        noise_stds = [2 ** p for p in range(-3, 6)]
        signal_amplitudes = [1.]
        nr_noises = [2 ** p for p in range(12)]
        if sim_type == ids.REGRESSION:
            nr_signals = [1]
        else:
            nr_signals = [2]

        return noise_stds, signal_amplitudes, nr_signals, nr_noises

    elif scan_type == ST_VARY_SIGNAL_NO_NOISE:
        noise_stds = [0]
        signal_amplitudes = [2 ** p for p in range(-6, 7)]
        nr_noises = [0]
        if sim_type == ids.REGRESSION:
            nr_signals = [2 ** p for p in range(7)]
        else:
            nr_signals = [2 ** p for p in range(1, 8)]

        return noise_stds, signal_amplitudes, nr_signals, nr_noises

    elif scan_type == ST_VARY_SIGNAL_HIGH_NOISE:
        noise_stds = [0]
        signal_amplitudes = [2 ** p for p in range(-5, 6)]
        nr_noises = [0]
        if sim_type == ids.REGRESSION:
            nr_signals = [2 ** p for p in range(7)]
        else:
            nr_signals = [2 ** p for p in range(1, 8)]

        return noise_stds, signal_amplitudes, nr_signals, nr_noises

    elif scan_type == ST_VARY_NOISE_AND_AMPLITUDE:
        noise_stds = [0]
        signal_amplitudes = [2 ** p for p in range(-5, 6)]
        nr_noises = [0]
        if sim_type == ids.REGRESSION:
            nr_signals = [2 ** p for p in range(7)]
        else:
            nr_signals = [2 ** p for p in range(1, 8)]

        return noise_stds, signal_amplitudes, nr_signals, nr_noises
    else:
        raise NotImplementedError(scan_type)


def scan_params(simulation_type: ids.SimulationType, optimizer: ids.OptimizerID) -> None:
    if simulation_type == ids.REGRESSION:
        nr_seeds = NR_SEEDS_SCAN_REGRESSION
    else:
        nr_seeds = NR_SEEDS_SCAN

    job_file = Path('jobs') / paths.SCAN_SUBDIR / f'job_{simulation_type}_{optimizer}.txt'
    with open(job_file, 'r') as f:
        jobs = f.readlines()

    # Connect with the database
    paths.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(databases.get_db_path(folder=paths.RESULT_DIR,
                                                 simulation_type=simulation_type,
                                                 optimizer=optimizer))
    data_table = databases.DATA_TABLE
    mean_table = databases.MEAN_TABLE
    var_table = databases.VARIANCE_TABLE
    col_keys = {}
    insert_cmd = {}
    for table in [data_table, mean_table, var_table]:
        col_keys[table]: list[str] = databases.get_all_cols(simulation_type=simulation_type, table=table)
        sql.make_table(conn=conn, table_name=table, col_names=col_keys[table], verbose=False)
        insert_cmd[table] = sql.get_insert_cmd(table=table, col_keys=col_keys[table])

    cur = conn.cursor()
    progress_tot = len(jobs)
    if simulation_type in ids.PRIMARY_SIMULATIONS:
        for progress, job in enumerate(jobs):
            print(f"\rScanning {100 * progress / progress_tot:0.1f}% completed", end="")
            params = job.strip().split(';')
            noise_std = float(params[0])
            signal_amplitude = float(params[1])
            nr_noise = int(params[2])
            nr_signal = int(params[3])
            for learning_rate in LEARNING_RATES:
                outcomes = np.zeros((nr_seeds, NR_TRIALS))
                for seed in range(nr_seeds):
                    outcomes[seed, :] = simulate(
                        simulation_type=simulation_type, optimizer_type=optimizer, learning_rate=learning_rate,
                        seed=seed, noise_std=noise_std, signal_amplitude=signal_amplitude, nr_noise=nr_noise,
                        nr_signal=nr_signal
                    )
                    insert_values = (noise_std, signal_amplitude, nr_noise, nr_signal, learning_rate, seed)
                    cur.execute(insert_cmd[data_table], insert_values + tuple(outcomes[seed, :]))
                    if simulation_type in ids.PERF_SIMULATIONS:
                        n_rows = NR_TRIALS - MAV_N + 1
                        n = outcomes[seed, :].strides[0]
                        outcome_2d = np.lib.stride_tricks.as_strided(outcomes, shape=(n_rows, MAV_N), strides=(n, n))
                        cur.execute(insert_cmd[var_table], insert_values + tuple(np.std(outcome_2d, axis=1)))
                insert_values = (noise_std, signal_amplitude, nr_noise, nr_signal, learning_rate)
                cur.execute(insert_cmd[mean_table], insert_values + tuple(np.mean(outcomes, axis=0)))
            conn.commit()
    else:
        raise NotImplementedError(simulation_type)
    cur.close()
    conn.close()
    os.remove(job_file)
    print(". Terminated successfully.")


def write_scan_jobs(scan_type: ScanType, reset: bool = False) -> None:
    folder = Path('jobs') / paths.SCAN_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    for simulation_type in ids.PRIMARY_SIMULATIONS:
        unique_cols = databases.get_unique_cols(simulation_type=simulation_type, table=databases.BEST_LR_TABLE)
        unique_values = {}
        for optimizer in ids.OPTIMIZERS:

            # Init database variables
            conn = sqlite3.connect(
                databases.get_db_path(folder=paths.RESULT_DIR, simulation_type=simulation_type, optimizer=optimizer)
            )
            if reset:
                for table in databases.ALL_TABLES:
                    sql.drop_table(conn=conn, table_name=table, verbose=False)
                    cols = databases.get_all_cols(simulation_type=simulation_type, table=table)
                    sql.make_table(conn=conn, table_name=table, col_names=cols, verbose=False)
            cur = conn.cursor()

            # Write a job file with all the parameter configurations to scan according to the scan type.
            if simulation_type in ids.PRIMARY_SIMULATIONS:
                with open(folder / f'job_{simulation_type}_{optimizer}.txt', 'w') as f:
                    noise_stds, signal_amplitudes, nr_noises, nr_signals = get_scan_params(
                        scan_type=scan_type, sim_type=simulation_type
                    )
                    for noise_std in noise_stds:
                        unique_values[keys.NOISE_STD] = noise_std
                        for signal_amplitude in signal_amplitudes:
                            unique_values[keys.SIGNAL_AMPLITUDE] = signal_amplitude
                            for nr_noise in nr_noises:
                                unique_values[keys.NUMBER_NOISE] = nr_noise
                                for nr_signal in nr_signals:
                                    unique_values[keys.NUMBER_SIGNAL] = nr_signal

                                    # Only write the job if it was not yet simulated
                                    if reset:
                                        write_job = True
                                    else:
                                        count = sql.count_rows(cur=cur, table=databases.BEST_LR_TABLE,
                                                               unique_values=unique_values, unique_cols=unique_cols)
                                        write_job = count == 0
                                    if write_job:
                                        f.write(f'{noise_std:.3E};{signal_amplitude:.3E};{nr_noise};{nr_signal}\n')
            else:
                raise NotImplementedError(simulation_type)
            cur.close()
            conn.close()


def update_best_lrs() -> None:
    # for simulation_type in ids.SIMULATION_TYPES:
    for simulation_type in ids.PRIMARY_SIMULATIONS:
        for optimizer in ids.OPTIMIZERS:
            db_path = databases.get_db_path(folder=paths.RESULT_DIR, simulation_type=simulation_type,
                                            optimizer=optimizer)
            cols = databases.get_all_cols(simulation_type=simulation_type, table=databases.BEST_LR_TABLE)
            if simulation_type in ids.LOSS_SIMULATIONS:
                best_train_lrs = sql.get_max(db_path=db_path, table=databases.MEAN_TABLE,
                                             max_col=f'{keys.LOSS}_{NR_TRIALS-1}', maxmin=False)
            elif simulation_type in ids.PERF_SIMULATIONS:
                best_train_lrs = sql.get_max(db_path=db_path, table=databases.MEAN_TABLE,
                                             max_col=f'{keys.PERFORMANCE}_{NR_TRIALS-1}', maxmin=True)
            else:
                raise NotImplementedError

            best_table = databases.BEST_LR_TABLE
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            sql.drop_table(conn=conn, table_name=best_table)
            sql.make_table(conn=conn, table_name=best_table, col_names=cols)
            insert_cmd = sql.get_insert_cmd(table=best_table, col_keys=cols)
            for best_params_set in best_train_lrs:
                cur.execute(insert_cmd, best_params_set)
            conn.commit()
            cur.close()
            conn.close()


if __name__ == '__main__':
    write_scan_jobs(scan_type=ST_VARY_SIGNAL_NO_NOISE)
    # update_best_lrs()
    # print(simulate(simulation_type=ids.TD_SALIENCE_NAIVE, learning_rate=0.1, seed=0, noise_std=1.,
    #                signal_amplitude=1., nr_noise=2, nr_signal=2, optimizer_type=ids.SGD, nr_noise_td=1,
    #                nr_signal_td=2))
    # scan_params(simulation_type=ids.REGRESSION, optimizer=ids.SGD)
    # scan_params(simulation_type=ids.GONOGO, optimizer=ids.SGD)
    # scan_params(simulation_type=ids.GONOGO_AVABS, optimizer=ids.SGD)
    # scan_params(simulation_type=ids.TD_SALIENCE_NAIVE, optimizer=ids.SGD)
    # scan_params(simulation_type=ids.REGRESSION, optimizer=ids.ADAM)
    # scan_params(simulation_type=ids.GONOGO, optimizer=ids.ADAM)
    # scan_params(simulation_type=ids.GONOGO_AVABS, optimizer=ids.ADAM)
    # scan_params(simulation_type=ids.TD_SALIENCE_NAIVE, optimizer=ids.ADAM)


