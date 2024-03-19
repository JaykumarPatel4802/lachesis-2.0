#convert .db sqlite3 file to .csv file
import sqlite3
import csv
import pandas as pd
import os


function_names = ['floatmatmul', 'videoprocess', 'imageprocess', 'linpack', 'encrypt']

def convert_db_to_csv(db_file):
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # for function_name in function_names: 
    #     cursor.execute("SELECT * FROM fxn_exec_data WHERE function=?", (function_name,))
    #     rows = cursor.fetchall()
    #     #get the column names
    #     cursor.execute("PRAGMA table_info(fxn_exec_data)")
    #     column_names = cursor.fetchall()
    #     #continue if there are no rows
    #     if len(rows) == 0:
    #         continue
    #     #write the column names to the csv file
    #     csv_dir = './data/' + function_name + '.csv'
    #     with open(csv_dir, 'w') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([i[1] for i in column_names])
    #         writer.writerows(rows)
    cursor.execute("SELECT * FROM fxn_exec_data")
    rows = cursor.fetchall()
    #get the column names
    cursor.execute("PRAGMA table_info(fxn_exec_data)")
    column_names = cursor.fetchall()
    #continue if there are no rows
    if len(rows) == 0:
        return
    #write the column names to the csv file
    file_name = os.path.splitext(db_file)[0]
    file_name = file_name.split('/')[-1]
    csv_dir = './data/' + file_name + '.csv'
    with open(csv_dir, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([i[1] for i in column_names])
        writer.writerows(rows)
        
def main():
    #go through all the .db files in sqlite directory and convert them to .csv files
    directory = './final_sqlite_data/'
    #get all the .db files in the directory
    # db_files = [f for f in os.listdir(directory) if f.endswith('.db')]
    # for db_file in db_files:
        # convert_db_to_csv(directory + db_file)
    
    convert_db_to_csv(directory + 'lachesis-controller-linpack-midway-snapshot.db')
    
    
    
if __name__ == '__main__':
    main()
    