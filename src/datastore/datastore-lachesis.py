import sqlite3
import os

def create_tables():
    
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('lachesis-controller.db')
    cursor = conn.cursor()

    # Create fxn_exec_data table
    cursor.execute('''CREATE TABLE IF NOT EXISTS fxn_exec_data (
                            function TEXT,
                            lachesis_start TEXT,
                            lachesis_end TEXT,
                            start_time TEXT,
                            end_time TEXT,
                            duration REAL,
                            cold_start_latency REAL,
                            slo REAL,
                            inputs TEXT,
                            cpu_limit REAL,
                            mem_limit REAL,
                            predicted_cpu REAL,
                            predicted_mem REAL,
                            p90_cpu REAL,
                            p95_cpu REAL,
                            p99_cpu REAL,
                            max_cpu REAL,
                            max_mem REAL,
                            invoker_ip TEXT,
                            invoker_name TEXT,
                            activation_id TEXT,
                            exp_no TEXT,
                            scheduled_cpu INTEGER,
                            scheduled_mem INTEGER,
                            energy REAL,
                            frequency REAL
                        )''')

    # Commit the changes and close the database connection
    conn.commit()

    # Create prediction_models table
    cursor.execute('''CREATE TABLE IF NOT EXISTS pred_models (
                        function TEXT,
                        host TEXT,
                        cpu_port TEXT,
                        mem_port TEXT,
                        no_invocations INTEGER
                  )''')
    conn.commit()
    conn.close()

# view all the rows in 'fxn_exec_data' table
def view_fxn_exec_table():
    conn = sqlite3.connect('lachesis-controller.db')
    cursor = conn.cursor()

    # Retrieve column names
    cursor.execute("PRAGMA table_info(fxn_exec_data)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Retrieve and print all rows from the table
    cursor.execute("SELECT * FROM fxn_exec_data")
    rows = cursor.fetchall()
    
    if not rows:
        print("No data found in the 'fxn_exec_data' table.")
    else:
        print("Column names:", columns)
        print("Data in the 'fxn_exec_data' table:")
        for row in rows:
            print(row)

    conn.close()

# view all the rows in 'pred_models_cpu' table
def view_pred_models():
    conn = sqlite3.connect('lachesis-controller.db')
    cursor = conn.cursor()

    # Retrieve column names
    cursor.execute("PRAGMA table_info(pred_models)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Retrieve and print all rows from the table
    cursor.execute("SELECT * FROM pred_models")
    rows = cursor.fetchall()
    
    if not rows:
        print("Column names:", columns)
        print("No data found in the 'pred_models' table.")
    else:
        print("Column names:", columns)
        print("Data in the 'pred_models' table:")
        for row in rows:
            print(row)

    conn.close()

# clear (delete all rows from) the 'function_executions' table
def clear_fxn_exec_table():
    conn = sqlite3.connect('lachesis-controller.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fxn_exec_data")
    conn.commit()
    conn.close()

# clear (delete all rows from) the 'pred_models_cpu' table
def clear_pred_models_table():
    conn = sqlite3.connect('lachesis-controller.db')
    cursor = conn.cursor()

    # Fetch all rows from the SQLite table
    cursor.execute("SELECT cpu_port, mem_port FROM pred_models")
    rows = cursor.fetchall()

    # Iterate through the rows and execute pkill command for each port value
    for row in rows:
        cpu_port = row[0]
        mem_port = row[1]
        os.system(f"pkill -9 -f 'vw.*--port {cpu_port}'")
        os.system(f"pkill -9 -f 'vw.*--port {mem_port}'")

    # Delete all rows from the table
    cursor.execute("DELETE FROM pred_models")
    conn.commit()
    
    # Close the database connection
    conn.close()

# delete 'function_executions' table
def delete_table(table):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('lachesis-controller.db')
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        table_exists = cursor.fetchone()

        if table_exists:
            # Delete the table
            cursor.execute(f"DROP TABLE '{table}'")
            conn.commit()
            print(f"The '{table}' table has been deleted.")
        else:
            print(f"The '{table}' table does not exist in the database.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the database connection
        if conn:
            conn.close()

# delete all rows except 
def delete_specific_exp(exp):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('lachesis-controller.db')
        cursor = conn.cursor()

        # Create a SQL command to delete rows where 'exp_no' is not 'exp_1'
        delete_query = f"DELETE FROM fxn_exec_data WHERE exp_no = '{exp}'"

        # Execute the delete query
        cursor.execute(delete_query)

        # Commit the changes
        conn.commit()

        # Close the database connection
        conn.close()

        # print("Rows deleted successfully, keeping only 'exp_1' rows.")
    except sqlite3.Error as e:
        print("SQLite error:", e)


if __name__=='__main__':
    # delete_table('fxn_exec_data')
    create_tables()
    # view_fxn_exec_table()
    # print()
    # view_pred_models()
    clear_fxn_exec_table()
    clear_pred_models_table()
    # delete_table('pred_models')
    # delete_specific_exp('lachesis-azure-rps-2')
    # delete_specific_exp('lachesis-azure-rps-4')
    # delete_specific_exp('lachesis-azure-rps-6')
    # delete_specific_exp('lachesis-packing-type-azure-quantile-0.9-slo-0.4-rps-2')
