import sqlite3
import os

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

if __name__=='__main__':
    # view_fxn_exec_table()
    view_pred_models()