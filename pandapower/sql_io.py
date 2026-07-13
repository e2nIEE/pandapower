# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from typing import Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

from pandapower import io_utils, pandapowerNet

try:
    import psycopg
    import psycopg.errors
    import psycopg.sql as psql

    PSYCOPG_INSTALLED = True
except ImportError:
    psycopg = None  # type: ignore[assignment]
    PSYCOPG_INSTALLED = False

try:
    import sqlite3

    SQLITE_INSTALLED = True
except ImportError:
    sqlite3 = None  # type: ignore[assignment]
    SQLITE_INSTALLED = False

import logging

if TYPE_CHECKING:
    import psycopg.sql as psql

logger = logging.getLogger(__name__)


def to_sql_str(string: str) -> "psql.Identifier":
    return psql.Identifier(*string.split('.'))


def match_sql_type(dtype):
    if dtype in ("float", "float32", "float64"):
        return "double precision"
    elif dtype in ("int", "int32", "int64", "uint32", "uint64", "Int64"):
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
    query = "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = %s);"
    cursor.execute(query, (table_name.split('.')[0], table_name.split('.')[-1]))
    (exists,) = cursor.fetchone()
    return exists


def get_sql_table_columns(cursor, table_name):
    query = "SELECT * FROM information_schema.columns WHERE table_schema = %s AND table_name = %s;"
    cursor.execute(query, (table_name.split('.')[0], table_name.split('.')[-1]))
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
        query = psql.SQL("SELECT * FROM {}").format(to_sql_str(table_name))
    else:
        columns_string = psql.SQL(' AND ').join(
            [psql.SQL("{} = {}").format(to_sql_str(k), v) for k, v in id_columns.items()])
        query = psql.SQL("SELECT * FROM {} WHERE {}").format(to_sql_str(table_name), columns_string)
    cursor.execute(query)
    colnames = [desc[0] for desc in cursor.description]
    table = cursor.fetchall()
    df = pd.DataFrame(table, columns=colnames)
    with pd.option_context('future.no_silent_downcasting', True):
        df = df.fillna(np.nan).infer_objects()
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
        tuples = [(*tuple(x), *id_columns.values()) for x in table[table_columns].itertuples(index=tuples_index)]
    else:
        tuples = [tuple(x) for x in table[table_columns].itertuples(index=tuples_index)]
    # Replace pd.NA values with None for conversion to postgres NULL
    tuples = [tuple(None if value is pd.NA else value for value in row) for row in tuples]

    # Comma-separated dataframe columns
    sql_columns = [index_name, *table_columns, *id_columns.keys()]
    sql_column_types = [index_type,
                        *[match_sql_type(t) for t in table[table_columns].dtypes.astype(str).values],
                        *[match_sql_type(np.result_type(type(v)).name) for v in id_columns.values()]]

    # check if all columns already exist and if not, add more columns
    existing_columns = get_sql_table_columns(cursor, table_name)
    new_columns = [(to_sql_str(c), t) for c, t in
                   zip(sql_columns, sql_column_types) if c not in existing_columns]
    if len(new_columns) > 0:
        logger.info(f"adding columns {new_columns} to table {table_name}")
        query = psql.SQL("ALTER TABLE {table} {add_columns};").format(
            table=to_sql_str(table_name),
            add_columns=psql.SQL(',').join([
                psql.SQL(f"ADD COLUMN {{column}} {type_}").format(column=col)
                for col, type_ in new_columns
            ])
        )
        cursor.execute(query)
        conn.commit()

    if timestamp:
        add_timestamp_column(conn, cursor, table_name)

    # SQL query to execute
    columns = [to_sql_str(c.replace('%', '%%')) for c in sql_columns]
    query = psql.SQL("INSERT INTO {tbl}({fields}) VALUES({placeholders})").format(
        tbl=to_sql_str(table_name),
        fields=psql.SQL(',').join(columns),
        placeholders=psql.SQL(',').join(psql.Placeholder() * len(sql_columns))
    )
    
    # batch_size = 1000
    # for chunk in tqdm(chunked(tuples, batch_size)):
    #     cursor.executemany(query, chunk)
    #     conn.commit()
    cursor.executemany(query, tuples)
    conn.commit()


def check_postgresql_catalogue_table(cursor, table_name, grid_id, grid_id_column, download=False):
    table_exists = check_if_sql_table_exists(cursor, table_name)

    if not table_exists:
        if download:
            raise UserWarning(f"grid catalogue {table_name} does not exist")
        else:
            query = psql.SQL(
                "CREATE TABLE {table_name}({column} BIGSERIAL PRIMARY KEY, timestamp TIMESTAMPTZ DEFAULT now());").format(
                table_name=to_sql_str(table_name),
                column=to_sql_str(grid_id_column)
            )
            cursor.execute(query)
    else:
        existing_columns = get_sql_table_columns(cursor, table_name)
        if grid_id_column not in existing_columns:
            raise UserWarning(f"grid_id_column {grid_id_column} is missing in grid catalogue {table_name}")
        if grid_id is None:
            if download:
                raise UserWarning(f"grid_id ({grid_id_column}) is None: {grid_id}")
            return  # we don't need to check for duplicates if grid_id is None (means we are uploading a new net)
        query = psql.SQL("SELECT COUNT(*) FROM {} where {}=%s").format(
            to_sql_str(table_name),
            to_sql_str(grid_id_column)
        )
        cursor.execute(query, (grid_id,))
        (found,) = cursor.fetchone()
        if download and found == 0:
            raise UserWarning(f"found no entries in {table_name} where {grid_id_column}={grid_id}")
        if not download and found > 0:
            raise UserWarning(f"found {found} duplicate entries in grid_catalogue where {grid_id_column}={grid_id}")


def create_postgresql_catalogue_entry(conn, cursor, grid_id, grid_id_column, catalogue_table_name):
    # check if a grid with the provided ids was already added
    check_postgresql_catalogue_table(cursor, catalogue_table_name, grid_id, grid_id_column)
    # create a "catalogue" table to keep track of all grids available in the DB
    if grid_id is None:
        query_str: psql.LiteralString = "INSERT INTO {catalogue}({column}) VALUES(DEFAULT) RETURNING {column}"
    else:
        query_str: psql.LiteralString = "INSERT INTO {catalogue}({column}) VALUES({value}) RETURNING {column}"
    query = psql.SQL(query_str).format(
        catalogue=to_sql_str(catalogue_table_name),
        column=to_sql_str(grid_id_column),
        value=None if grid_id is None else to_sql_str(grid_id),
    )
    cursor.execute(query)
    conn.commit()
    (written_grid_id,) = cursor.fetchone()
    return written_grid_id


def add_timestamp_column(conn, cursor, table_name):
    cursor.execute(
        psql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ;").format(to_sql_str(table_name)))
    conn.commit()
    cursor.execute(psql.SQL("ALTER TABLE {} ALTER COLUMN timestamp SET DEFAULT now();").format(to_sql_str(table_name)))
    conn.commit()


def create_sql_table_if_not_exists(conn, cursor, table_name, grid_id_column, catalogue_table_name):
    query = psql.SQL(
        "CREATE TABLE IF NOT EXISTS {table}({column} BIGINT, FOREIGN KEY({column}) REFERENCES {catalogue}({column}) ON DELETE CASCADE);").format(
        table=to_sql_str(table_name),
        column=to_sql_str(grid_id_column),
        catalogue=to_sql_str(catalogue_table_name),
    )
    cursor.execute(query)
    conn.commit()


def delete_postgresql_net(
        grid_id: int,
        dsn: str,
        schema: str,
        grid_id_column: str = "grid_id",
        grid_catalogue_name: str = "grid_catalogue",
) -> None:
    """
    Removes a grid model from the PostgreSQL database.

    Parameters:
        grid_id: unique grid_id that will be used to identify the data for the grid model
        dsn: data source name according to pep-249
        schema: name of the database schema (e.g. 'postgres')
        grid_id_column: name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
        grid_catalogue_name: name of the catalogue table that includes all grid_id values and the timestamp when the
            grid data were added

    Examples:
        >>> delete_postgresql_net(0, "postgresql://user:password@host:port/database", "test_schema", "grid_id", "grid_catalogue")
    """
    if not PSYCOPG_INSTALLED:
        raise UserWarning("install the package psycopg to use PostgreSQL I/O in pandapower")

    with psycopg.connect(conninfo=dsn) as conn: # type: ignore[union-attr]
        cursor = conn.cursor()
        catalogue_table_name = grid_catalogue_name if schema is None else f"{schema}.{grid_catalogue_name}"
        check_postgresql_catalogue_table(cursor, catalogue_table_name, grid_id, grid_id_column, download=True)
        query = psql.SQL("DELETE FROM {} WHERE {}=%s;").format(
            to_sql_str(catalogue_table_name),
            to_sql_str(grid_id_column),
        )
        cursor.execute(query, (grid_id,))
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
        except psycopg.errors.UndefinedTable as err:
            logger.info(f"skipped {element} due to error: {err}")
            continue

        if 'geo' in tab.columns:
            tab.geo = tab.geo.replace({'NaN': None})

        d[element] = tab

    net = io_utils.from_dict_of_dfs(d, net=empty_dict_like_object)

    return net


def to_sql(net, conn, schema, include_results=False, grid_id=None, grid_id_column="grid_id",
           grid_catalogue_name="grid_catalogue", index_name=None):
    """
    Uploads a pandapowerNet to a PostgreSQL database. The database must exist, the element tables
    are created if they do not exist.
    TODO: JSON serialization (e.g. for controller objects) is not implemented yet.

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
        dodfs = {}
        for t, in cursor.fetchall():
            table = pd.read_sql_query('SELECT * FROM "{}"'.format(t.replace('"', '""')), conn, index_col="index")
            table.index.name = None
            dodfs[t] = table
        net = io_utils.from_dict_of_dfs(dodfs)
    return net


def to_postgresql(
        net: pandapowerNet,
        dsn: str,
        schema: str,
        include_results: bool = False,
        grid_id: Optional[int] = None,
        grid_id_column: str = "grid_id",
        grid_catalogue_name: str = "grid_catalogue",
        index_name=None,
    ) -> int:
    """
    Uploads a pandapowerNet to a PostgreSQL database. The database must exist, the element tables
    are created if they do not exist.
    JSON serialization (e.g. for controller objects) is not implemented yet.

    Parameters:
        net: the grid model to be uploaded to the database
        dsn: data source name according to pep-249
        schema: name of the database schema (e.g. 'postgres')
        include_results: specify whether the power flow results are included when the grid is uploaded
        grid_id: unique grid_id that will be used to identify the data for the grid model, default None.
            If None, it will be set automatically by PostgreSQL
        grid_id_column: name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
        grid_catalogue_name: name of the catalogue table that includes all grid_id values and the timestamp when
            the grid data were added
        index_name: name of the custom column to be used inplace of index in the element tables if it is not the
            standard DataFrame index

    Returns:
        either the user-specified grid_id or the automatically generated grid_id of the grid model
    """
    if not PSYCOPG_INSTALLED:
        raise UserWarning("install the package psycopg to use PostgreSQL I/O in pandapower")
    logger.debug(f"Uploading the grid data to the DB schema {schema}")

    with psycopg.connect(dsn) as conn: # type: ignore[union-attr]
        grid_id = to_sql(net, conn, schema, include_results, grid_id, grid_id_column, grid_catalogue_name, index_name)
    return grid_id


def from_postgresql(
        grid_id: int,
        dsn: str,
        schema: str,
        grid_id_column: str = "grid_id",
        grid_catalogue_name: str = "grid_catalogue",
        empty_dict_like_object: Optional[dict] = None,
        grid_tables = None,
):
    """
    Downloads an existing pandapowerNet from a PostgreSQL database.

    Parameters:
        grid_id: unique grid_id that will be used to identify the data for the grid model
        dsn: data source name according to pep-249
        schema: name of the database schema (e.g. 'postgres')
        grid_id_column: name of the column for "grid_id" in the PosgreSQL tables, default="grid_id".
        grid_catalogue_name: name of the catalogue table that includes all grid_id values and the timestamp when
            the grid data were added
        empty_dict_like_object: If None, the output of pandapower.create_empty_network() is used as an empty element
            to be filled by the grid data.
            Give another dict-like object to start filling that alternative object with the data.
        grid_tables:

    Returns:
        the loaded pandapower network
    """
    if not PSYCOPG_INSTALLED:
        raise UserWarning("install the package psycopg to use PostgreSQL I/O in pandapower")

    with psycopg.connect(dsn) as conn: # type: ignore[union-attr]
        net = from_sql(conn, schema, grid_id, grid_id_column, grid_catalogue_name, empty_dict_like_object, grid_tables)

    return net
