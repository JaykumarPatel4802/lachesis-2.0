import sqlite3

db_file = "invoker_data.db"

def create_function_energy_utilization_advanced_table(cursor):
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS function_energy_utilization_advanced (
            container_id TEXT,
            timestamp TEXT,
            socket REAL,
            duration_sec REAL,
            num_proc REAL,
            num_threads REAL,
            pkg_credit_frac REAL,
            dram_credit_frac REAL,
            total_pkg_joules REAL,
            total_dram_joules REAL,
            base_pkg_joules REAL,
            base_dram_joules REAL,
            ascribed_pkg_joules REAL,
            ascribed_dram_joules REAL,
            tracer_pkg_joules REAL,
            tracer_dram_joules REAL,
            pkg_percent REAL,
            dram_percent REAL
        );
    """
    cursor.execute(create_table_sql)

def create_function_executions_table(cursor):
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS function_executions (
            tid TEXT,
            start_time TEXT,
            end_time TEXT,
            function TEXT,
            container_id TEXT,
            duration INTEGER,
            cold_start_latency INTEGER,
            pk INTEGER,
            activation_id TEXT
        );
        """        
    cursor.execute(create_table_sql)

def create_function_utilization_advanced_table(cursor):
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS function_utilization_advanced (
            container_id TEXT,
            timestamp TEXT,
            cpu_usage_ns REAL,
            num_cores REAL,
            curr_system_usage REAL,
            mem_util REAL,
            mem_limit REAL
        );
        """        
    cursor.execute(create_table_sql)

def main():
    try:
        # Open the database
        print(f"Opening {db_file}")
        db = sqlite3.connect(db_file)
        
        # Set WAL mode
        print("Setting WAL mode")
        result = db.execute("PRAGMA journal_mode=WAL;").fetchone()
        print("Journal set to: ", result)
        db.commit()

        cursor = db.cursor()

        # To delete the table and reset all data
        # db.execute("DROP TABLE function_energy_utilization_advanced")
        
        create_function_utilization_advanced_table(cursor)
        create_function_executions_table(cursor)
        create_function_energy_utilization_advanced_table(cursor)
        
        db.commit()  # Commit the changes
        print("Done with DB stuff")
        
    except sqlite3.Error as e:
        print("SQLite error:", e)
    except Exception as e:
        print("General error: ", e)

if __name__=='__main__':
    main()