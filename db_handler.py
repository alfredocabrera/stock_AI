import sqlite3
from sqlite3 import Error


def db_close(conn):
    conn.close()


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def db_connect():
    database = "db/ai.db"

    sql_create_tasks_table = """CREATE TABLE IF NOT EXISTS tasks (
                                    id integer AUTO_INCREMENT PRIMARY KEY,
                                    current_price decimal NOT NULL,
                                    predicted_price decimal NOT NULL,
                                    current_date date NOT NULL,
                                    predicted_date date NOT NULL
                                );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create tasks table
        create_table(conn, sql_create_tasks_table)
    else:
        print("Error! cannot create the database connection.")

    return conn




