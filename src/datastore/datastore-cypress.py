import sqlite3
import os

def create_tables():
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('cypress-controller.db')
    cursor = conn.cursor()

    # Create fxn_exec_data table
    cursor.execute('''CREATE TABLE IF NOT EXISTS fxn_exec_data (
                            function TEXT,
                            start_time TEXT,
                            end_time TEXT,
                            duration REAL,
                            cold_start_latency REAL,
                            slo REAL,
                            inputs TEXT,
                            cpu_limit REAL,
                            mem_limit REAL,
                            p90_cpu REAL,
                            p95_cpu REAL,
                            p99_cpu REAL,
                            max_cpu REAL,
                            max_mem REAL,
                            invoker_ip TEXT,
                            invoker_name TEXT,
                            activation_id TEXT,
                            exp_no TEXT
                        )''')

    # Commit the changes and close the database connection
    conn.commit()
    conn.close()

# Function to view all rows in specified table
def view_table(table):
    table_info_sql = f'PRAGMA table_info({table})'
    rows_sql = f'SELECT * FROM {table}'

    conn = sqlite3.connect('cypress-controller.db')
    cursor = conn.cursor()

    # Retrieve column names
    cursor.execute(table_info_sql)
    columns = [column[1] for column in cursor.fetchall()]
    
    # Retrieve and print all rows from the table
    cursor.execute(rows_sql)
    rows = cursor.fetchall()
    
    if not rows:
        print(f"No data found in the '{table}' table.")
    else:
        print("Column names:", columns)
        print(f"Data in the '{table}' table:")
        for row in rows:
            print(row)

    conn.close()


if __name__=='__main__':
    create_tables()
    view_table('fxn_exec_data')