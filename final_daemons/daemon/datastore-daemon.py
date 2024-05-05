import sqlite3

# Function to create the 'function_executions' and 'function_utilization' tables
def create_tables():
    
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('invoker_data.db')
    cursor = conn.cursor()

    # Create the 'function_executions' table
    cursor.execute('''CREATE TABLE IF NOT EXISTS function_executions (
                        tid TEXT PRIMARY KEY,
                        start_time TEXT,
                        end_time TEXT,
                        function TEXT,
                        container_id TEXT,
                        duration INTEGER,
                        cold_start_latency INTEGER,
                        pk INTEGER,
                        activation_id TEXT
                    )''')

    # Commit the changes and close the database connection
    conn.commit()

    # Create the 'function_utilization' table
    cursor.execute(''' CREATE TABLE IF NOT EXISTS function_utilization (
                        container_id TEXT,
                        timestamp TEXT,
                        cpu_util REAL,
                        mem_util REAL,
                        mem_limit REAL
                   )''')
    conn.commit()

    # Create the 'function_utilization' table
    cursor.execute(''' CREATE TABLE IF NOT EXISTS function_utilization_advanced (
                        container_id TEXT,
                        timestamp TEXT,
                        cpu_usage_ns REAL, 
                        num_cores REAL, 
                        curr_system_usage REAL,
                        mem_util REAL,
                        mem_limit REAL
                   )''')
    conn.commit()

    # Create the 'slo_execs' table
    cursor.execute('''CREATE TABLE IF NOT EXISTS slo_execs (
                        tid TEXT PRIMARY KEY,
                        start_time TEXT,
                        end_time TEXT,
                        function TEXT,
                        container_id TEXT,
                        duration INTEGER,
                        cold_start_latency INTEGER,
                        cpu INTEGER,
                        inputs TEXT,
                        p90_cpu REAL,
                        p95_cpu REAL,
                        p99_cpu REAL,
                        max_cpu REAL,
                        max_mem REAL,
                        invoker_ip TEXT
                    )''')
    conn.close()

# Function to view all rows in specified table
def view_table(table):
    table_info_sql = f'PRAGMA table_info({table})'
    rows_sql = f'SELECT * FROM {table}'

    conn = sqlite3.connect('invoker_data.db')
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

# Function clear (delete all rows from) the specified table
def clear_table(table):
    delete_sql = f'DELETE FROM {table}'
    conn = sqlite3.connect('invoker_data.db')
    cursor = conn.cursor()
    cursor.execute(delete_sql)
    conn.commit()
    conn.close()


if __name__=='__main__':
    create_tables()
    # view_table('function_executions')
    # view_table('function_utilization')
    # view_table('function_utilization_advanced')
    # view_table('slo_execs')
    clear_table('slo_execs')
    clear_table('function_executions')
    clear_table('function_utilization')
    clear_table('function_utilization_advanced')


