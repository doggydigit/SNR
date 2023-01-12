from utils import keys
import sqlite3
from pathlib import Path
from typing import List, Tuple
from os.path import dirname
from time import sleep as time_sleep
from numpy.random import RandomState

DATA_TYPE_TEXT = "TEXT"
DATA_TYPE_REAL = "REAL"
DATA_TYPE_INT = "INT"

NR_ATTEMPTS_BEFORE_GIVE_UP = 20
SLEEP_TIME = 31.4159265359


def connect(db_path: Path) -> sqlite3.Connection:
    Path(dirname(db_path)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def sleep():
    prng = RandomState()
    time_sleep(SLEEP_TIME + prng.uniform(low=0., high=SLEEP_TIME))
    del prng


def disconnect(conn: sqlite3.Connection) -> None:
    conn.close()
    del conn


def get_data_type(column: str) -> str:

    if column == keys.NUMBER_NOISE:
        return DATA_TYPE_INT
    elif column == keys.NUMBER_SIGNAL:
        return DATA_TYPE_INT
    elif column == keys.NOISE_STD:
        return DATA_TYPE_REAL
    elif column == keys.SIGNAL_AMPLITUDE:
        return DATA_TYPE_REAL
    elif column == keys.NUMBER_NOISE_TD:
        return DATA_TYPE_INT
    elif column == keys.NUMBER_SIGNAL_TD:
        return DATA_TYPE_INT
    elif column == keys.LEARNING_RATE:
        return DATA_TYPE_REAL
    elif column == keys.LEARNING_RATE_TD:
        return DATA_TYPE_REAL
    elif column == keys.SEED:
        return DATA_TYPE_INT
    else:
        if keys.LOSS in column:
            return DATA_TYPE_REAL
        if keys.PERFORMANCE in column:
            return DATA_TYPE_REAL
        else:
            raise NotImplementedError(column)


def make_table(conn: sqlite3.Connection, table_name: str, col_names: List[str], verbose: bool = False) -> None:
    cur = conn.cursor()
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='" + table_name + "';")
    if cur.fetchone()[0] != 1:
        cmd = "CREATE TABLE {} (".format(table_name)
        cmd += ", ".join([cn + " " + get_data_type(cn) for cn in col_names])
        cmd += ");"
        conn.execute(cmd)
        if verbose:
            print("Table {} was created".format(table_name))
    elif verbose:
        print("Table {} already exists".format(table_name))
    conn.commit()
    cur.close()
    del cur


def drop_table(conn: sqlite3.Connection, table_name: str, verbose: bool = True) -> None:
    cur = conn.cursor()
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='" + table_name + "';")
    if cur.fetchone()[0] == 1:
        cmd = "DROP TABLE {};".format(table_name)
        conn.execute(cmd)
        if verbose:
            print("Table {} was dropped".format(table_name))

    elif verbose:
        print("Table {} doesn't exist and could therefore not be dropped".format(table_name))


def get_insert_cmds(table: str, values: dict):

    columns = ",".join(values.keys())
    placeholders = ", ".join("?" * len(values.values()))
    insert_cmd = "INSERT INTO {}({}) VALUES({})".format(table, columns, placeholders)
    insert_values = tuple(values.values())

    return insert_cmd, insert_values


def insert(db_path: Path, table: str, unique_values: dict, unique_cols: List[str], outcome_values: dict,
           outcome_cols: List[str]) -> None:

    # Make sure all the required values are defined
    assert all(c in unique_values.keys() for c in unique_cols)
    assert all(c in outcome_values.keys() for c in outcome_cols)

    # Prepare the sqlite commands before executing them to reduce the interaction time with the database
    insert_cmd, insert_values = get_insert_cmds(table=table, values={**unique_values, **outcome_values})

    # Connect with the database
    conn = connect(db_path=db_path)
    cur = conn.cursor()

    cur.execute(insert_cmd, insert_values)

    # Commit changes and disconnect
    conn.commit()
    cur.close()
    del cur
    disconnect(conn=conn)


def get_where_cmd(unique_cols: List[str]) -> str:
    return 'WHERE (' + ' = ? AND '.join(unique_cols) + ' = ?)'


def get_exists_cmd(table: str, unique_values: dict, unique_cols: List[str]) -> Tuple[str, tuple]:
    where_cmd = get_where_cmd(unique_cols=unique_cols)
    exists_cmd = "SELECT count(*) FROM {} {};".format(table, where_cmd)
    exists_values = tuple(unique_values[k] for k in unique_cols)
    return exists_cmd, exists_values


def count_rows(cur: sqlite3.Cursor, table: str, unique_values: dict, unique_cols: List) -> int:

    # Write the command out that does the query
    where_cmd, where_values = get_exists_cmd(table=table, unique_values=unique_values, unique_cols=unique_cols)

    # Execute query
    cur.execute(where_cmd, where_values)

    return cur.fetchone()[0]


def get_insert_cmd(table: str, col_keys: list[str]) -> str:
    columns = ','.join(col_keys)
    placeholders = ', '.join('?' * len(col_keys))
    insert_cmd = 'INSERT INTO {}({}) VALUES({})'.format(table, columns, placeholders)
    return insert_cmd


def get_upsert_cmds(table: str, unique_values: dict, unique_cols: List[str], outcome_values: dict,
                    outcome_cols: List[str]) -> Tuple[str, tuple, str, tuple, str, tuple]:

    # Get the rows with the
    exists_cmd, exists_values = get_exists_cmd(table=table, unique_values=unique_values, unique_cols=unique_cols)

    update_cmd = ""
    outcomes = []
    for outcome_col in outcome_cols:
        update_cmd += " {} = ?,".format(outcome_col)
        outcomes += [outcome_values[outcome_col]]
    update_cmd = "UPDATE {} SET{} {}".format(table, update_cmd[:-1], get_where_cmd(unique_cols=unique_cols))
    update_values = tuple(outcomes + [unique_values[k] for k in unique_cols])

    insert_cmd, insert_values = get_insert_cmds(table=table, values={**unique_values, **outcome_values})

    return exists_cmd, exists_values, update_cmd, update_values, insert_cmd, insert_values


def get_upsert_cmd(conn: sqlite3.Connection, table: str, unique_values: dict, unique_cols: List[str],
                   outcome_values: dict, outcome_cols: List[str]) -> tuple[str, tuple]:

    # Prepare the sqlite commands before executing them to reduce the interaction time with the database
    where_cmd, where_values, update_cmd, update_values, insert_cmd, insert_values = get_upsert_cmds(
        table=table, unique_values=unique_values, unique_cols=unique_cols, outcome_values=outcome_values,
        outcome_cols=outcome_cols
    )

    # Check if row already exists
    cur = conn.cursor()
    cur.execute(where_cmd, where_values)

    # Either update or add performance
    if cur.fetchone()[0] == 0:
        cmd = (insert_cmd, insert_values)
    else:
        cmd = (update_cmd, update_values)

    cur.close()
    del cur

    return cmd


def upsert(db_path: Path, table: str, unique_values: dict, unique_cols: List[str], outcome_values: dict,
           outcome_cols: List[str]) -> None:

    # Make sure all the required values are defined
    assert all(c in unique_values.keys() for c in unique_cols)
    assert all(c in outcome_values.keys() for c in outcome_cols)

    # Connect with the database
    conn = connect(db_path=db_path)

    # get the right value for an upsert
    cmd, values = get_upsert_cmd(conn=conn, table=table, unique_values=unique_values, unique_cols=unique_cols,
                                 outcome_values=outcome_values, outcome_cols=outcome_cols)

    # Upsert
    cur = conn.cursor()
    cur.execute(cmd, values)

    # Commit changes and disconnect
    conn.commit()
    cur.close()
    del cur
    disconnect(conn=conn)


def delete(conn: sqlite3.Connection, table: str, values: dict, unique_cols: List[str]) -> None:
    cmd = "DELETE FROM {} {}".format(table, get_where_cmd(unique_cols=unique_cols))
    conn.execute(cmd, tuple(values[k] for k in unique_cols))


def select(db_path: Path, table: str, get_cols: List[str], where_values: dict | tuple) -> List[Tuple]:
    conn = connect(db_path=db_path)
    cur = conn.cursor()
    where_keys = [str(k) for k in where_values.keys()]
    cmd = "SELECT {} FROM {} {}".format(", ".join(get_cols), table, get_where_cmd(unique_cols=where_keys))
    where_tuple = tuple(where_values[k] for k in where_keys)
    cur.execute(cmd, where_tuple)
    rows = cur.fetchall()
    cur.close()
    disconnect(conn=conn)
    return rows


def get_update_avg_cmd(data_table: str, mean_table: str, avg_cols: List[str], unique_cols: List[str]) -> List[str]:
    cmds = []
    for avg_col in avg_cols:
        where_cmd = " AND ".join(["{}.{} = {}.{}".format(data_table, col, mean_table, col) for col in unique_cols])
        cmds += ["UPDATE {} SET {} = (SELECT AVG({}) FROM {} WHERE {});"
                 "".format(mean_table, avg_col, avg_col, data_table, where_cmd)]
    return cmds


def get_max(db_path: Path, table: str, max_col: str, maxmin: bool = True):
    if maxmin:
        maxmin_str = "max"
    else:
        maxmin_str = "min"
    group_by = ", ".join([keys.NOISE_STD, keys.NUMBER_NOISE, keys.NUMBER_SIGNAL])
    select_cols = ", ".join([group_by, keys.LEARNING_RATE])
    conn = connect(db_path=db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT {select_cols}, {maxmin_str}({max_col}) FROM {table} GROUP BY {group_by};")
    max_rows = cur.fetchall()
    cur.close()
    disconnect(conn=conn)
    return max_rows
