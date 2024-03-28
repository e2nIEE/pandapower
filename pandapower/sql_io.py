# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
import numpy as np

from pandapower import io_utils

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.errors

    PSYCOPG2_INSTALLED = True
except ImportError:
    psycopg2 = None
    PSYCOPG2_INSTALLED = False

try:
    import sqlite3

    SQLITE_INSTALLED = True
except ImportError:
    sqlite3 = None
    SQLITE_INSTALLED = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def match_sql_type(dtype):
    if dtype in ("float", "float32", "float64"):
        return "double precision"
    elif dtype in ("int", "int32", "int64", "uint32", "uint64"):
        return "bigint"
    elif dtype in ("object", "str"):
        return "varchar"
    elif dtype == "bool":
        return "boolean"
    elif "datetime" in dtype:
        return "timestamp"
    else:
        raise UserWarning(f"unsupported type {dtype}")


def check_if_sql_table_exists(cursor, table_name):
    query = f"SELECT EXISTS (SELECT FROM information_schema.tables " \
            f"WHERE table_schema = '{table_name.split('.')[0]}' " \
            f"AND table_name = '{table_name.split('.')[-1]}');"
    cursor.execute(query)
    (exists,) = cursor.fetchone()
    return exists


def get_sql_table_columns(cursor, table_name):
    query = f"SELECT * FROM information_schema.columns " \
            f"WHERE table_schema = '{table_name.split('.')[0]}' " \
            f"AND table_name   = '{table_name.split('.')[-1]}';"
    cursor.execute(query)
    colnames = [desc[0] for desc in cursor.description]
    list_idx = colnames.index("column_name")
    columns_data = cursor.fetchall()
    columns = [c[list_idx] for c in columns_data]
    return columns


def download_sql_table(cursor, table_name, **id_columns):
    # first we check if table exists:
    exists = check_if_sql_table_exists(cursor, table_name)
    if not exists:
        raise UserWarning(f"table {table_name} does not exist or the user has no access to it")

    if len(id_columns.keys()) == 0:
        query = f"SELECT * FROM {table_name}"
    else:
        columns_string = ' and '.join([f"{str(k)} = '{str(v)}'" for k, v in id_columns.items()])
        query = f"SELECT * FROM {table_name} WHERE {columns_string}"

    cursor.execute(query)
    colnames = [desc[0] for desc in cursor.description]
    table = cursor.fetchall()
    df = pd.DataFrame(table, columns=colnames)
    df = df.fillna(np.nan)
    index_name = f"{table_name.split('.')[-1]}_id"
    if index_name in df.columns:
        df = df.set_index(index_name)
    if len(id_columns) > 0:
        df.drop(id_columns.keys(), axis=1, inplace=True)
    return df


def upload_sql_table(conn, cursor, table_name, table, index_name=None, timestamp=False, **id_columns):
    # index_name allows using a custom column for the table index and disregard the DataFrame index,
    # otherwise a <table_name>_id is used as index_name and DataFrame index is also uploaded to the database
    table = table.where(pd.notnull(table), None)
    if index_name is None:
        index_name = f"{table_name.split('.')[-1]}_id"
        index_type = match_sql_type(str(table.index.dtype))
        table_columns = [c for c in table.columns if c not in id_columns]
        tuples_index = True
    else:
        index_type = match_sql_type(str(table[index_name].dtype))
        table_columns = [c for c in table.columns if c != index_name and c not in id_columns]
        tuples_index = False

    # Create a list of tuples from the dataframe values
    if len(id_columns.keys()) > 0:
        tuples = [(*tuple(x), *id_columns.values())
                  for x in table[table_columns].itertuples(index=tuples_index)]
    else:
        tuples = [tuple(x) for x in table[table_columns].itertuples(index=tuples_index)]

    # Comma-separated dataframe columns
    sql_columns = [index_name, *table_columns, *id_columns.keys()]
    sql_column_types = [index_type,
                        *[match_sql_type(t) for t in table[table_columns].dtypes.astype(str).values],
                        *[match_sql_type(np.result_type(type(v)).name) for v in id_columns.values()]]
    placeholders = ",".join(['%s'] * len(sql_columns))

    # check if all columns already exist and if not, add more columns
    existing_columns = get_sql_table_columns(cursor, table_name)
    new_columns = [('"%s"' % c, t) for c, t in zip(sql_columns, sql_column_types) if c not in existing_columns]
    if len(new_columns) > 0:
        logger.info(f"adding columns {new_columns} to table {table_name}")
        column_statement = ", ".join(f"ADD COLUMN {c} {t}" for c, t in new_columns)
        query = f"ALTER TABLE {table_name} {column_statement};"
        cursor.execute(query)
        conn.commit()

    if timestamp:
        add_timestamp_column(conn, cursor, table_name)

    # SQL query to execute
    columns = ['"%s"' % c for c in sql_columns]
    query = f"INSERT INTO {table_name}({','.join(columns)}) VALUES({placeholders})"
    # batch_size = 1000
    # for chunk in tqdm(chunked(tuples, batch_size)):
    #     cursor.executemany(query, chunk)
    #     conn.commit()
    psycopg2.extras.execute_batch(cursor, query, tuples, page_size=100)
    conn.commit()


def check_postgresql_catalogue_table(cursor, table_name, grid_id, grid_id_column, download=False):
    table_exists = check_if_sql_table_exists(cursor, table_name)

    if not table_exists:
        if download:
            raise UserWarning(f"grid catalogue {table_name} does not exist")
        else:
            query = f"CREATE TABLE {table_name} ({grid_id_column} BIGSERIAL PRIMARY KEY, " \
                    f"timestamp TIMESTAMPTZ DEFAULT now());"
            cursor.execute(query)
    else:
        existing_columns = get_sql_table_columns(cursor, table_name)
        if grid_id_column not in existing_columns:
            raise UserWarning(f"grid_id_column {grid_id_column} is missing in grid catalogue {table_name}")
        if grid_id is None:
            if download:
                raise UserWarning(f"grid_id ({grid_id_column}) is None: {grid_id}")
            return  # we don't need to check for duplicates if grid_id is None (means we are uploading a new net)
        query = f"SELECT COUNT(*) FROM {table_name} where {grid_id_column}={grid_id}"
        cursor.execute(query)
        (found,) = cursor.fetchone()
        if download and found == 0:
            raise UserWarning(f"found no entries in {table_name} where {grid_id_column}={grid_id}")
        if not download and found > 0:
            raise UserWarning(f"found {found} duplicate entries in grid_catalogue where {grid_id_column}={grid_id}")


def create_postgresql_catalogue_entry(conn, cursor, grid_id, grid_id_column, catalogue_table_name):
    # check if a grid with the provided ids was already added
    check_postgresql_catalogue_table(cursor, catalogue_table_name, grid_id, grid_id_column)
    # create a "catalogue" table to keep track of all grids available in the DB
    query = f"INSERT INTO {catalogue_table_name}({grid_id_column}) VALUES({'DEFAULT' if grid_id is None else grid_id}) " \
            f"RETURNING {grid_id_column}"
    cursor.execute(query)
    conn.commit()
    (written_grid_id,) = cursor.fetchone()
    return written_grid_id


def add_timestamp_column(conn, cursor, table_name):
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ;")
    conn.commit()
    cursor.execute(f"ALTER TABLE {table_name} ALTER COLUMN timestamp SET DEFAULT now();")
    conn.commit()


def create_sql_table_if_not_exists(conn, cursor, table_name, grid_id_column, catalogue_table_name):
    query = f"CREATE TABLE IF NOT EXISTS {table_name}({grid_id_column} BIGINT, " \
            f"FOREIGN KEY({grid_id_column}) REFERENCES {catalogue_table_name}({grid_id_column})" \
            f"ON DELETE CASCADE);"
    cursor.execute(query)
    conn.commit()


def delete_postgresql_net(grid_id, host, user, password, database, schema, grid_id_column="grid_id",
                          grid_catalogue_name="grid_catalogue"):
    """
    Removes a grid model from the PostgreSQL database.

    Parameters
    ----------
    grid_id : int
        unique grid_id that will be used to identify the data for the grid model
    host : str
        hostname for the DB, e.g. "localhost"
    user : str
    password : str
    database : str
        name of the database
    schema : str
        name of the database schema (e.g. 'postgres')
    grid_id_column : str
        name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that includes all grid_id values and the timestamp when the grid data were added

    Returns
    -------

    """
    if not PSYCOPG2_INSTALLED:
        raise UserWarning("install the package psycopg2 to use PostgreSQL I/O in pandapower")

    conn = psycopg2.connect(host=host, user=user, password=password, database=database)
    cursor = conn.cursor()
    catalogue_table_name = grid_catalogue_name if schema is None else f"{schema}.{grid_catalogue_name}"
    check_postgresql_catalogue_table(cursor, catalogue_table_name, grid_id, grid_id_column, download=True)
    query = f"DELETE FROM {catalogue_table_name} WHERE {grid_id_column}={grid_id};"
    cursor.execute(query)
    conn.commit()


def from_sql(conn, schema, grid_id, grid_id_column="grid_id", grid_catalogue_name="grid_catalogue",
             empty_dict_like_object=None, grid_tables=None):
    """
    Downloads an existing pandapowerNet from a PostgreSQL database.

    Parameters
    ----------
    conn : connection to SQL database (e.g. SQLite, PostgreSQL)
    schema : str
        name of the database schema (e.g. 'postgres')
    grid_id : int
        unique grid_id that will be used to identify the data for the grid model
    grid_id_column : str
        name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that includes all grid_id values and the timestamp when the grid data were added
    empty_dict_like_object : dict-like
        If None, the output of pandapower.create_empty_network() is used as an empty element to be filled by
        the grid data. Give another dict-like object to start filling that alternative object with the data.

    Returns
    -------
    net : pandapowerNet
    """
    cursor = conn.cursor()
    id_columns = {grid_id_column: grid_id}
    if grid_tables is None:
        catalogue_table_name = grid_catalogue_name if schema is None else f"{schema}.{grid_catalogue_name}"
        check_postgresql_catalogue_table(cursor, catalogue_table_name, grid_id, grid_id_column, download=True)
        grid_tables = download_sql_table(cursor, "grid_tables" if schema is None else f"{schema}.grid_tables", **id_columns)

    d = {}
    for element in grid_tables.table.values:
        table_name = element if schema is None else f"{schema}.{element}"
        try:
            tab = download_sql_table(cursor, table_name, **id_columns)
        except UserWarning as err:
            logger.debug(err)
            continue
        except psycopg2.errors.UndefinedTable as err:
            logger.info(f"skipped {element} due to error: {err}")
            continue

        d[element] = tab

    net = io_utils.from_dict_of_dfs(d, net=empty_dict_like_object)

    return net


def to_sql(net, conn, schema, include_results=False, grid_id=None, grid_id_column="grid_id",
           grid_catalogue_name="grid_catalogue", index_name=None):
    """
    Uploads a pandapowerNet to a PostgreSQL database. The database must exist, the element tables
    are created if they do not exist.
    JSON serialization (e.g. for controller objects) is not implemented yet.

    Parameters
    ----------
    net : pandapowerNet
        the grid model to be uploaded to the database
    conn : connection to SQL database (e.g. SQLite, PostgreSQL)
    schema : str
        name of the database schema (e.g. 'postgres')
    include_results : bool
        specify whether the power flow results are included when the grid is uploaded, default=False
    grid_id : int
        unique grid_id that will be used to identify the data for the grid model, default None.
        If None, it will be set automatically by PostgreSQL
    grid_id_column : str
        name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that includes all grid_id values and the timestamp when the grid data were added
    index_name : str
        name of the custom column to be used inplace of index in the element tables if it is not the standard DataFrame index

    Returns
    -------
    grid_id: int
        returns either the user-specified grid_id or the automatically generated grid_id of the grid model
    """
    cursor = conn.cursor()
    catalogue_table_name = grid_catalogue_name if schema is None else f"{schema}.{grid_catalogue_name}"
    d = io_utils.to_dict_of_dfs(net, include_results=include_results, include_empty_tables=False)
    written_grid_id = create_postgresql_catalogue_entry(conn, cursor, grid_id, grid_id_column, catalogue_table_name)
    id_columns = {grid_id_column: written_grid_id}
    d["grid_tables"] = pd.DataFrame(d.keys(), columns=["table"])
    for element, element_table in d.items():
        table_name = element if schema is None else f"{schema}.{element}"
        # None causes postgresql error, np.nan is better
        create_sql_table_if_not_exists(conn, cursor, table_name, grid_id_column, catalogue_table_name)
        upload_sql_table(conn=conn, cursor=cursor, table_name=table_name, table=element_table,
                         index_name=index_name, **id_columns)
        logger.debug(f"uploaded table {element}")
    return written_grid_id


def to_sqlite(net, filename, include_results=False):
    """
    Saves pandapowerNet an SQLite format

    Parameters
    ----------
    net : grid model
        pandapowerNet
    filename : path to a text file where the data will be stored
        str
    include_results : whether result tables should be included
        bool
    """
    if not SQLITE_INSTALLED:
        raise UserWarning("sqlite3 is not installed, install sqlite3 to use from_sqlite()")
    with sqlite3.connect(filename) as conn:
        dodfs = io_utils.to_dict_of_dfs(net, include_results=include_results)
        for name, data in dodfs.items():
            data.to_sql(name, conn)


def from_sqlite(filename):
    """
    Loads a grid model from SQLite format

    Parameters
    ----------
    filename : path to the text file where the data are stored

    Returns
    -------
    net : the grid model
        pandapowerNet
    """
    if not SQLITE_INSTALLED:
        raise UserWarning("sqlite3 is not installed, install sqlite3 to use from_sqlite()")
    with sqlite3.connect(filename) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        dodfs = dict()
        for t, in cursor.fetchall():
            table = pd.read_sql_query("SELECT * FROM '%s'" % t, conn, index_col="index")
            table.index.name = None
            dodfs[t] = table
        net = io_utils.from_dict_of_dfs(dodfs)
    return net


def to_postgresql(net, host, user, password, database, schema, include_results=False,
                  grid_id=None, grid_id_column="grid_id", grid_catalogue_name="grid_catalogue", index_name=None):
    """
    Uploads a pandapowerNet to a PostgreSQL database. The database must exist, the element tables
    are created if they do not exist.
    JSON serialization (e.g. for controller objects) is not implemented yet.

    Parameters
    ----------
    net : pandapowerNet
        the grid model to be uploaded to the database
    host : str
        hostname for connecting to the database
    user : str
        username for logging in
    password : str
    database : str
        name of the database
    schema : str
        name of the database schema (e.g. 'postgres')
    include_results : bool
        specify whether the power flow results are included when the grid is uploaded, default=False
    grid_id : int
        unique grid_id that will be used to identify the data for the grid model, default None.
        If None, it will be set automatically by PostgreSQL
    grid_id_column : str
        name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that includes all grid_id values and the timestamp when the grid data were added
    index_name : str
        name of the custom column to be used inplace of index in the element tables if it is not the standard DataFrame index

    Returns
    -------
    grid_id: int
        returns either the user-specified grid_id or the automatically generated grid_id of the grid model
    """
    if not PSYCOPG2_INSTALLED:
        raise UserWarning("install the package psycopg2 to use PostgreSQL I/O in pandapower")
    logger.debug(f"Uploading the grid data to the DB schema {schema}")
    with psycopg2.connect(host=host, user=user, password=password, database=database) as conn:
        grid_id = to_sql(net, conn, schema, include_results, grid_id, grid_id_column, grid_catalogue_name, index_name)
    return grid_id


def from_postgresql(grid_id, host, user, password, database, schema, grid_id_column="grid_id",
                    grid_catalogue_name="grid_catalogue", empty_dict_like_object=None, grid_tables=None):
    """
    Downloads an existing pandapowerNet from a PostgreSQL database.

    Parameters
    ----------
    grid_id : int
        unique grid_id that will be used to identify the data for the grid model
    host : str
        hostname for connecting to the database
    user : str
        username for logging in
    password : str
    database : str
        name of the database
    schema : str
        name of the database schema (e.g. 'postgres')
    grid_id_column : str
        name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
    grid_catalogue_name : str
        name of the catalogue table that includes all grid_id values and the timestamp when the grid data were added
    empty_dict_like_object : dict-like
        If None, the output of pandapower.create_empty_network() is used as an empty element to be filled by
        the grid data. Give another dict-like object to start filling that alternative object with the data.

    Returns
    -------
    net : pandapowerNet
    """
    if not PSYCOPG2_INSTALLED:
        raise UserWarning("install the package psycopg2 to use PostgreSQL I/O in pandapower")

    with psycopg2.connect(host=host, user=user, password=password, database=database) as conn:
        net = from_sql(conn, schema, grid_id, grid_id_column, grid_catalogue_name, empty_dict_like_object, grid_tables)

    return net
