import warnings
from pathlib import Path
from sqlite3 import OperationalError
from utils import ids, keys, sql
from utils.params import NR_TRIALS

MAV_N = 10
NR_MAV_TRIALS = NR_TRIALS - MAV_N + 1
DATA_TABLE = "data"
MEAN_TABLE = "mean_seed"
VARIANCE_TABLE = "var_seed"
BEST_LR_TABLE = "best_lr"
ALL_TABLES = [DATA_TABLE, MEAN_TABLE, VARIANCE_TABLE, BEST_LR_TABLE]


def get_db_path(folder: Path, simulation_type: ids.SimulationType, optimizer: ids.OptimizerID) -> Path:
    return folder / f'scan_{simulation_type}_{optimizer}.db'

# def get_unique_cols_scan(model_id: ids.ModelID, mean_seed: Optional[bool] = False, pool_tasks: Optional[bool] = False
#                          ) -> List[UniqueColumnType]:
#     """
#     Returns the list of columns names that uniquely define an entry into the database storing the outcomes from
#     hyperparameter scans, depending on the model id. These include training parameters and network architecture.
#     Parameters
#     ----------
#     model_id : ModelID
#         Identifier for the mode type
#     mean_seed : bool
#         Whether the columns are from the table with averages over seeds (i.e. seeds are not included in the columns)
#     pool_tasks : bool
#         Whether the table pools multiple different single task performances (and the task should therefore be included)
#     """
#
#     if mean_seed:
#         suffix = []
#     else:
#         suffix = [keys.K_SEED]
#
#     if model_id == ids.ID_SINGLE_TASK:
#         if pool_tasks:
#             prefix = [keys.K_TRAIN_TASKS]
#         else:
#             prefix = []
#         return prefix + [keys.K_ARCHITECTURE, keys.K_BATCH_SIZE, keys.K_SHARED_PARAM_LEARNING_RATE] + suffix
#     else:
#         return [keys.K_ARCHITECTURE, keys.K_BATCH_SIZE, keys.K_SHARED_PARAM_LEARNING_RATE,
#                 keys.K_CONTEXT_PARAM_LEARNING_RATE] + suffix


def get_unique_cols(simulation_type: ids.SimulationType, table: str) -> list[keys.Key]:

    if simulation_type in ids.PRIMARY_SIMULATIONS:
        if table in [DATA_TABLE, VARIANCE_TABLE]:
            suffix = [keys.LEARNING_RATE, keys.SEED]
        elif table == MEAN_TABLE:
            suffix = [keys.LEARNING_RATE]
        elif table == BEST_LR_TABLE:
            suffix = []
        else:
            raise ValueError(table)
        return [keys.NOISE_STD, keys.SIGNAL_AMPLITUDE, keys.NUMBER_NOISE, keys.NUMBER_SIGNAL] + suffix
    elif simulation_type in [ids.TD_SALIENCE, ids.TD_VALUE]:
        if table in [DATA_TABLE, VARIANCE_TABLE]:
            suffix = [keys.LEARNING_RATE_TD, keys.SEED]
        elif table == MEAN_TABLE:
            suffix = [keys.LEARNING_RATE_TD]
        elif table == BEST_LR_TABLE:
            suffix = []
        else:
            raise ValueError(table)
        return [keys.NOISE_STD, keys.SIGNAL_AMPLITUDE, keys.NUMBER_NOISE, keys.NUMBER_SIGNAL, keys.LEARNING_RATE,
                keys.NUMBER_SIGNAL_TD, keys.NUMBER_NOISE_TD] + suffix
    else:
        raise NotImplementedError


# def get_unique_values(prog_params: typeddicts.ProgramParameters, unique_cols: List[UniqueColumnType],
#                       train_params: Optional[typeddicts.TrainParameters] = None) -> dict:
#     """
#     Returns the values of the columns uniquely identifying an outcome database entry.
#     Parameters
#     ----------
#     prog_params : ProgramParameters
#         Program parameters of the simulation
#     train_params : TrainParameters
#         Training parameters of the simulation
#     unique_cols : List[str]
#         List of unique column keys
#     """
#     unique_values = {}
#     for c in keys.PROGRAM_KEYS:
#         if c in unique_cols:
#             if c == keys.K_ARCHITECTURE:
#                 unique_values[c] = parameters.program.architecture_list_to_str(prog_params[c])
#             elif c == keys.K_TRAIN_TASKS:
#                 unique_values[c] = parameters.program.tasks_list_to_str(prog_params[c])
#             else:
#                 unique_values[c] = prog_params[c]
#
#     for c in keys.TRAIN_KEYS:
#         if c in unique_cols:
#             unique_values[c] = train_params[c]
#
#     return unique_values


def get_outcome_cols(simulation_type: ids.SimulationType, table: str) -> list[str]:
    if table in [DATA_TABLE, MEAN_TABLE]:
        if simulation_type in ids.LOSS_SIMULATIONS:
            return [f"{keys.LOSS}_{n}" for n in range(NR_TRIALS)]
        elif simulation_type in ids.PERF_SIMULATIONS:
            return [f"{keys.PERFORMANCE}_{n}" for n in range(NR_TRIALS)]
        else:
            raise NotImplementedError
    elif table == VARIANCE_TABLE:
        if simulation_type in ids.LOSS_SIMULATIONS:
            return [f"{keys.LOSS}_{n}" for n in range(NR_MAV_TRIALS)]
        elif simulation_type in ids.PERF_SIMULATIONS:
            return [f"{keys.PERFORMANCE}_{n}" for n in range(NR_MAV_TRIALS)]
        else:
            raise NotImplementedError
    elif table == BEST_LR_TABLE:
        if simulation_type in ids.LOSS_SIMULATIONS:
            return [keys.LOSS]
        elif simulation_type in ids.PERF_SIMULATIONS:
            return [keys.PERFORMANCE]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError(table)


def get_all_cols(simulation_type: ids.SimulationType, table: str) -> list[str]:
    all_cols = get_unique_cols(simulation_type=simulation_type, table=table)
    all_cols += get_outcome_cols(simulation_type=simulation_type, table=table)
    return all_cols

# def is_outcome_simulated(prog_params: typeddicts.ProgramParameters, train_params: typeddicts.TrainParameters,
#                          save_params: typeddicts.SavingParameters) -> bool:
#     """
#     Returns whether the performance outcome for the given set of simulation parameters has already been simulated
#     and saved into the performance outcome database.
#     """
#
#     # Connect with the database
#     db_path = save_params[keys.K_OUTCOME_DIR] / parameters.saving.get_perf_db_name(
#         prog_name=prog_params[keys.K_PROGRAM_NAME])
#
#     # Get the table name
#     table = parameters.saving.get_table(model_id=prog_params[keys.K_MODEL_ID],
#                                         task=parameters.program.get_single_task(prog_params))
#
#     # Get the columns and values that define the simulation
#     unique_cols = get_unique_cols(prog_name=prog_params[keys.K_PROGRAM_NAME], model_id=prog_params[keys.K_MODEL_ID])
#     unique_values = get_unique_values(prog_params=prog_params, train_params=train_params, unique_cols=unique_cols)
#
#     n_rows = sql.count_rows(db_path=db_path, table=table, unique_values=unique_values, unique_cols=unique_cols)
#
#     outcome_was_simulated = n_rows >= 1
#     if n_rows >= 2:
#         warnings.warn("You have duplicate rows in db {} in table {}. Think about deduping".format(db_path, table))
#
#     return outcome_was_simulated
#
#
# def init_performance_table(prog_params: typeddicts.ProgramParameters, save_params: typeddicts.SavingParameters) -> None:
#     """
#     If the simulation requires to saving performance outcome, this function will check whether the table where to saving
#     these outcomes already exists and create it, if it doesn't exist yet.
#     Parameters
#     ----------
#     prog_params : ProgramParameters
#     save_params : SavingParameters
#     """
#     if save_params[keys.K_SAVE_PERFORMANCE]:
#         unique_cols = get_unique_cols(prog_name=prog_params[keys.K_PROGRAM_NAME], model_id=prog_params[keys.K_MODEL_ID])
#         cols: List[Union[keys.ProgramKey, keys.TrainKey, keys.PerfKey, keys.LossKey]] = unique_cols
#         outcome_cols = get_outcome_cols(prog_name=prog_params[keys.K_PROGRAM_NAME], performance=True)
#         cols = cols + outcome_cols
#         task = parameters.program.get_single_task(prog_params=prog_params)
#         table = parameters.saving.get_table(model_id=prog_params[keys.K_MODEL_ID], task=task)
#         sql.make_table(db_path=save_params[keys.K_PERFORMANCE_PATH], table_name=table, col_names=cols)
#
#
# def init_pmdd_table(prog_params: typeddicts.ProgramParameters, save_params: typeddicts.SavingParameters) -> None:
#     """
#     If the simulation requires to saving tracked pmdd loss, this function will check whether the table where to saving
#     these outcomes already exists and create it, if it doesn't exist yet.
#     Parameters
#     ----------
#     prog_params : ProgramParameters
#     save_params : SavingParameters
#     """
#
#     if save_params[keys.K_SAVE_PMDD_LOSS]:
#         unique_cols = get_unique_cols(prog_name=prog_params[keys.K_PROGRAM_NAME], model_id=prog_params[keys.K_MODEL_ID])
#         cols: List[Union[keys.ProgramKey, keys.TrainKey, keys.PerfKey, keys.LossKey]] = unique_cols
#
#         outcome_cols = get_outcome_cols(prog_name=prog_params[keys.K_PROGRAM_NAME], performance=False)
#         cols = cols + outcome_cols
#         table = parameters.saving.get_table(model_id=prog_params[keys.K_MODEL_ID],
#                                             task=parameters.program.get_single_task(prog_params))
#
#         for db_name in [paths.DB_PMDD_FIRST_EPOCH_BATCHES_NAME, paths.DB_PMDD_TRAIN_EPOCHS_NAME]:
#             db_path = save_params[keys.K_TRACK_PMDD_DIR] / db_name
#             sql.make_table(db_path=db_path, table_name=table, col_names=cols)
#
#
# def initialize_simulation(prog_params: typeddicts.ProgramParameters, save_params: typeddicts.SavingParameters,
#                           train_params: Optional[typeddicts.TrainParameters] = None, verbose: Optional[bool] = False
#                           ) -> ParameterSetType:
#
#     # Initialize all the parameters necessary to the program, the training and the storing of outcomes are well-defined.
#     prog_params = parameters.program.init_prog_params(prog_params=prog_params)
#     train_params = parameters.training.init_training_params(prog_params=prog_params, train_params=train_params,
#                                                             verbose=verbose)
#     save_params = parameters.saving.init_save_params(prog_params=prog_params, save_params=save_params,
#                                                      train_params=train_params)
#
#     # Make sure the table where to store the network's performance outcome is defined
#     if not prog_params[keys.K_STRONGLY_PARALLEL]:
#         if save_params[keys.K_SAVE_PERFORMANCE]:
#             failed_connect = True
#             for attempt in range(sql.NR_ATTEMPTS_BEFORE_GIVE_UP):
#                 try:
#                     saving.databases.init_performance_table(prog_params=prog_params, save_params=save_params)
#                 except OperationalError:
#                     sql.sleep()
#                 else:
#                     failed_connect = False
#                     break
#             if failed_connect:
#                 raise TimeoutError
#
#         if save_params[keys.K_SAVE_PMDD_LOSS]:
#             failed_connect = True
#             for attempt in range(sql.NR_ATTEMPTS_BEFORE_GIVE_UP):
#                 try:
#                     saving.databases.init_pmdd_table(prog_params=prog_params, save_params=save_params)
#                 except OperationalError:
#                     sql.sleep()
#                 else:
#                     failed_connect = False
#                     break
#             if failed_connect:
#                 raise TimeoutError
#     return prog_params, train_params, save_params
#
#
# def init_outcome_tables(prog_name: ids.ProgramName, model_ids: List[ids.ModelID], datasets: List[ids.DatasetID],
#                         pretrain_dataset: Optional[ids.DatasetID] = None) -> None:
#     test_perf = ids.ID_SCAN_HYPERPARAMS not in prog_name
#     pmdd_save = (ids.ID_SCAN_HYPERPARAMS not in prog_name) and (ids.ID_TRAIN in prog_name)
#
#     for model_id in model_ids:
#         for dataset_id in datasets:
#             if model_id == ids.ID_SINGLE_TASK:
#                 tasks = list(range(data.datasets.get_number_tasks(dataset=dataset_id)))
#             else:
#                 tasks = [0]
#             for task in tasks:
#
#                 # Program parameters
#                 prog_params: typeddicts.ProgramParameters = {keys.K_PROGRAM_NAME: prog_name,
#                                                              keys.K_MODEL_ID: model_id,
#                                                              keys.K_ARCHITECTURE: [100],
#                                                              keys.K_DATASET: dataset_id,
#                                                              keys.K_SEEDS: [0],
#                                                              keys.K_TRAIN_TASKS: [task],
#                                                              keys.K_STRONGLY_PARALLEL: False}
#                 if pretrain_dataset is not None:
#                     prog_params[keys.K_PRETRAIN_DATASET] = pretrain_dataset
#
#                 # Saving parameters
#                 save_params: typeddicts.SavingParameters = {keys.K_SAVE_PERFORMANCE: True,
#                                                             keys.K_TEST_PERFORMANCE: test_perf,
#                                                             keys.K_SAVE_INIT_MODEL_PARAMS: False,
#                                                             keys.K_SAVE_FINAL_MODEL_PARAMS: False,
#                                                             keys.K_SAVE_PMDD_LOSS: pmdd_save}
#
#                 # Fully initialize all parameters and tables
#                 initialize_simulation(prog_params=prog_params, save_params=save_params, verbose=False)
#
#
# def drop_col(cols: List[ColumnType], col_to_drop: ColumnType) -> List[ColumnType]:
#     """
#     Removes the first occurrence of the designed column from the list of column names given
#     Parameters
#     ----------
#     cols : List[ColumnType]
#         List of columns from which to remove the specified column
#     col_to_drop : ColumnType
#         Name of the column to remove
#     """
#     for i in range(len(cols)):
#         if cols[i] == col_to_drop:
#             cols.pop(i)
#             return cols
#     warnings.warn("The column '{}' could n0t be dropped as it was not present in the list".format(col_to_drop))
#     return cols
#
#
# def update_seed_avg(prog_name: ids.ProgramName, model_id: ids.ModelID, db_path: Path,
#                     performance: bool = True, task: Optional[int] = 0) -> None:
#
#     # The columns that should be averaged
#     outcome_cols = get_outcome_cols(prog_name=prog_name, performance=performance)
#
#     # The columns for which averages should be found (with the seed key removed)
#     unique_cols = get_unique_cols_scan(model_id=model_id)
#     unique_cols = drop_col(cols=unique_cols, col_to_drop=keys.K_SEED)
#
#     # Make sure the table to store seed averages exists
#     table_cols = unique_cols + outcome_cols
#     mean_table = parameters.saving.get_table(model_id=model_id, task=task, mean_seed=True)
#     sql.drop_table(db_path=db_path, table_name=mean_table)
#     sql.make_table(db_path=db_path, table_name=mean_table, col_names=table_cols)
#
#     # Name of table containing outcomes to average
#     data_table = parameters.saving.get_table(model_id=model_id, task=task)
#
#     # Write the sql commands to update the seed averages
#     update_cmds = sql.get_update_avg_cmd(data_table=data_table, mean_table=mean_table,
#                                          avg_cols=outcome_cols,
#                                          unique_cols=unique_cols)
#
#     # Connect to database
#     conn = sql.connect(db_path=db_path)
#     cur = conn.cursor()
#
#     # Make sure the rows of average performance outcomes exist in the K_MEAN_SEED table (or create them)
#     distinct_cols = ", ".join(unique_cols)
#     get_cmd = "SELECT DISTINCT {} FROM {}".format(distinct_cols, data_table)
#     cur.execute(get_cmd)
#     rows = cur.fetchall()
#     for row in rows:
#         values = {}
#         for c in range(len(unique_cols)):
#             values[unique_cols[c]] = row[c]
#         exists_cmd, exists_values = sql.get_exists_cmd(table=mean_table, unique_values=values,
#                                                        unique_cols=unique_cols)
#         cur.execute(exists_cmd, exists_values)
#         if cur.fetchone()[0] == 0:
#             for outcome_col in outcome_cols:
#                 values[outcome_col] = -666.666
#             insert_cmd, insert_values = sql.get_insert_cmds(table=mean_table, values=values)
#             cur.execute(insert_cmd, insert_values)
#     conn.commit()
#
#     # Execute sql commands
#     for cmd in update_cmds:
#         cur.execute(cmd)
#     conn.commit()
#     cur.close()
#     sql.disconnect(conn=conn)
#
#
# def update_task_avg(prog_params: typeddicts.ProgramParameters, performance: Optional[bool] = True) -> None:
#
#     # Reset/empty the table where to pool the scan outcomes for all single tasks (pre-averaged over seeds)
#     all_tasks_dir = parameters.saving.get_outcome_dir(prog_params=prog_params, mean_task=True)
#     all_tasks_db = all_tasks_dir / paths.DB_HYPERPARAMS_PERF_NAME
#     pool_task_table = parameters.saving.get_table(model_id=ids.ID_SINGLE_TASK, mean_seed=True, pool_task=True)
#     pool_tasks_unique_cols: List[ColumnType] = get_unique_cols(
#         prog_name=prog_params[keys.K_PROGRAM_NAME], model_id=ids.ID_SINGLE_TASK, mean_seed=True, pool_tasks=True)
#     outcome_cols = get_outcome_cols(prog_name=prog_params[keys.K_PROGRAM_NAME], performance=performance)
#     pool_tasks_cols = pool_tasks_unique_cols + outcome_cols
#     sql.drop_table(db_path=all_tasks_db, table_name=pool_task_table)
#     sql.make_table(db_path=all_tasks_db, table_name=pool_task_table, col_names=pool_tasks_cols)
#
#     # Column names of interest to pool all the individual task data
#     task_col = keys.K_TRAIN_TASKS
#     mean_task_unique_cols: List[ColumnType] = get_unique_cols(prog_name=prog_params[keys.K_PROGRAM_NAME],
#                                                               model_id=ids.ID_SINGLE_TASK, mean_seed=True)
#     col_str = ",".join(mean_task_unique_cols + outcome_cols)
#
#     # Pool the single task outcomes for all tasks (averaged over seeds) into a single (freshly emptied) table
#     for task in range(data.datasets.get_number_tasks(dataset=prog_params[keys.K_DATASET])):
#         prog_params[keys.K_TRAIN_TASKS] = [task]
#         single_task_dir = parameters.saving.get_outcome_dir(prog_params=prog_params, mean_task=False)
#         single_task_db = single_task_dir / paths.DB_HYPERPARAMS_PERF_NAME
#         single_task_table = parameters.saving.get_table(model_id=ids.ID_SINGLE_TASK, task=task, mean_seed=True)
#         from_conn = sql.connect(db_path=single_task_db)
#         cur = from_conn.cursor()
#         cur.execute("ATTACH DATABASE '{}' AS new_db;".format(all_tasks_db))
#         cur.execute("INSERT INTO new_db.{} ({},{}) SELECT {},{} FROM {};".format(
#             pool_task_table, task_col, col_str, task, col_str, single_task_table))
#         from_conn.commit()
#         cur.close()
#         del cur
#         sql.disconnect(conn=from_conn)
#
#     # Make sure the table to store task averages exists
#     mean_task_table = parameters.saving.get_table(model_id=ids.ID_SINGLE_TASK, mean_seed=True, mean_task=True)
#     mean_task_cols = mean_task_unique_cols + outcome_cols
#     sql.make_table(db_path=all_tasks_db, table_name=mean_task_table, col_names=mean_task_cols)
#
#     # Write the sql commands to update the task averages
#     update_cmds = sql.get_update_avg_cmd(data_table=pool_task_table, mean_table=mean_task_table,
#                                          avg_cols=outcome_cols, unique_cols=mean_task_unique_cols)
#
#     # Connect to database
#     conn = sql.connect(db_path=all_tasks_db)
#     cur = conn.cursor()
#
#     # Make sure the rows of average performance outcomes exist in the K_MEAN_TASK table (or create them)
#     distinct_cols = ", ".join(mean_task_unique_cols)
#     get_cmd = "SELECT DISTINCT {} FROM {}".format(distinct_cols, pool_task_table)
#     cur.execute(get_cmd)
#     rows = cur.fetchall()
#     for row in rows:
#         values = {}
#         for c in range(len(mean_task_unique_cols)):
#             values[mean_task_unique_cols[c]] = row[c]
#         exists_cmd, exists_values = sql.get_exists_cmd(table=mean_task_table, unique_values=values,
#                                                        unique_cols=mean_task_unique_cols)
#         cur.execute(exists_cmd, exists_values)
#         if cur.fetchone()[0] == 0:
#             for outcome_col in outcome_cols:
#                 values[outcome_col] = -666.666
#             insert_cmd, insert_values = sql.get_insert_cmds(table=mean_task_table, values=values)
#             cur.execute(insert_cmd, insert_values)
#         else:
#             raise NotImplementedError
#     conn.commit()
#
#     # Execute sql commands
#     for cmd in update_cmds:
#         cur.execute(cmd)
#     conn.commit()
#     cur.close()
#     sql.disconnect(conn=conn)
#
#
# def dedupe(prog_params, save_params):
#     # Get the columns and values that define the simulation
#     unique_cols = get_unique_cols(prog_name=prog_params[keys.K_PROGRAM_NAME], model_id=prog_params[keys.K_MODEL_ID])
#
#     # Get the table name
#     table = parameters.saving.get_table(model_id=prog_params[keys.K_MODEL_ID],
#                                         task=parameters.program.get_single_task(prog_params))
#
#     # Write the command out that does the de-duplication
#     where_cmd = " AND ".join(["{}.{} = p2.{}".format(table, c, c) for c in unique_cols])
#     delete_cmd = "DELETE FROM {} WHERE EXISTS (SELECT 1 FROM {} p2 WHERE {} AND {}.rowid < p2.rowid);".format(
#         table, table, where_cmd, table)
#
#     # Connect with the database
#     db_path = save_params[keys.K_OUTCOME_DIR] / parameters.saving.get_perf_db_name(
#         prog_name=prog_params[keys.K_PROGRAM_NAME])
#     conn = sql.connect(db_path=db_path)
#     cur = conn.cursor()
#
#     # Execute delete command
#     cur.execute(delete_cmd)
#     conn.commit()
#
#     # Disconnect from database
#     cur.close()
#     del cur
#     sql.disconnect(conn=conn)
#
#
# # TODO: This function can be deleted once you checked that st scans are now save in the correct db
# def move_table(dataset: ids.DatasetID):
#     cols = [keys.K_ARCHITECTURE, keys.K_BATCH_SIZE, keys.K_SHARED_PARAM_LEARNING_RATE,
#             keys.K_SEED, keys.K_TRAIN_LOSS, keys.K_TRAIN_PERFORMANCE, keys.K_VALIDATION_LOSS,
#             keys.K_VALIDATION_PERFORMANCE]
#     from_db = paths.SCAN_HYPERPARAMS_RESULT_DIR / dataset / ids.ID_SINGLE_TASK / paths.DB_HYPERPARAMS_PERF_NAME
#     tasks = list(range(data.datasets.get_number_tasks(dataset=dataset)))
#     for task in tasks:
#         table = parameters.saving.get_table(model_id=ids.ID_SINGLE_TASK, task=task, mean_seed=False)
#         to_db = paths.SCAN_HYPERPARAMS_RESULT_DIR / dataset / table / paths.DB_HYPERPARAMS_PERF_NAME
#         sql.make_table(db_path=to_db, table_name=table, col_names=cols, verbose=True)
#         from_conn = sql.connect(db_path=from_db)
#         from_cur = from_conn.cursor()
#         # print("ATTACH DATABASE '{}' AS new_db;".format(to_db))
#         from_cur.execute("ATTACH DATABASE '{}' AS new_db;".format(to_db))
#         from_cur.execute("INSERT INTO new_db.{} SELECT * FROM {};".format(table, table))
#         from_conn.commit()
#         from_cur.close()
#         del from_cur
#         sql.disconnect(conn=from_conn)
#
#
# # TODO: This function can be deleted once you checked that st dirs are clean an work well in the new way
# def clean_tables(dataset: ids.DatasetID) -> None:
#     # db_path = paths.SCAN_HYPERPARAMS_RESULT_DIR / dataset / ids.ID_SINGLE_TASK / paths.DB_HYPERPARAMS_PERF_NAME
#     db_path = paths.TRAIN_FULL_DATASET_RESULT_DIR / dataset / paths.DB_FINAL_PERF_NAME
#     tasks = list(range(data.datasets.get_number_tasks(dataset=dataset)))
#     # for table in ids.MODEL_IDS:
#     #     sql.drop_table(db_path=db_path, table_name=table)
#     for task in tasks:
#         table = parameters.saving.get_table(model_id=ids.ID_SINGLE_TASK, task=task, mean_seed=False)
#         sql.drop_table(db_path=db_path, table_name=table)
#         table = parameters.saving.get_table(model_id=ids.ID_SINGLE_TASK, task=task, mean_seed=True)
#         sql.drop_table(db_path=db_path, table_name=table)
