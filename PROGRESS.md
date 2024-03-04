# Progress on Measurement Study
- Successfully Register Floatmatmult with memory 5120 and 1-32 cpu
- Successfully invoke the function from controller and get readings

# Next Steps
- Need to look into the following bug from `aggregator-daemon.py`
```
Traceback (most recent call last):
  File "/home/cc/daemon/aggregator-daemon.py", line 327, in <module>
    monitor_start_lines(db_conn_main, cursor_main, data_queue)
  File "/home/cc/daemon/aggregator-daemon.py", line 91, in monitor_start_lines
    insert_execution_data(log_line, cs_tid, cs_latency, db_conn, cursor)
  File "/home/cc/daemon/aggregator-daemon.py", line 68, in insert_execution_data
    cursor.execute("INSERT INTO function_executions VALUES (?, ?, NULL, ?, ?, NULL, ?, ?, ?)",
sqlite3.IntegrityError: UNIQUE constraint failed: function_executions.tid
```