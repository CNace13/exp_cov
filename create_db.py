import sqlite3
import random
from threading import Thread
from multiprocessing import Process, Queue, cpu_count
# from queue import Queue

from util.data_util import *

THREADS = 6
ROWS = 5000 # Number of rows to generate
TRANSACTION_SIZE = 50000
BATCH_SIZE = 500

# Database INSERT that supports SMT
def database_writer(db_path, table_name, data_util, result_queue):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    total_rows_written = 0
    while True:
        try:
            data_chunk = result_queue.get(timeout=1)
            if data_chunk is None:  # Sentinel value to indicate end of data
                break
            for i, data_row in enumerate(data_chunk):
                if total_rows_written % TRANSACTION_SIZE == 0:
                    if total_rows_written > 0: # All but the first
                        conn.commit()  # Commit the current transaction
                    cursor.execute("BEGIN TRANSACTION")
                    print('Writing', total_rows_written, data_row)
                
                cursor.execute(f"INSERT OR IGNORE INTO {table_name} (id, {', '.join(data_util.columns)}) VALUES (?, {', '.join(['?' for _ in data_util.columns])})", [data_row[0]] + data_row[1])
                total_rows_written += 1
        except:
            print('Queue empty')
    
    conn.commit()  # Commit any remaining transactions
    conn.close()


def create_database(db_path, table_name, data_util):
    """
    Creates an SQLite database with a table and populates it based on the possible values and specified distributions.
    
    Args:
    db_path (str): Path to the SQLite database file.
    table_name (str): Name of the table to create.
    data_util (DataUtil): ClassObj that helps create random data
    rows (int): The number of rows to have the data_util generate and place in the db
    
    Returns:
    None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    columns_def = ", ".join([f"{col} INTEGER" for col in data_util.columns])
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id PRIMARY KEY, {columns_def})"
    cursor.execute(create_table_query)

    conn.commit()
    conn.close()

    # Begin generating data and inserting into DB

    # Queue for storing generated data
    result_queue = Queue()
    threads = []
    
    # Create and start the database writer thread
    writer_thread = Thread(target=database_writer, args=(db_path, table_name, data_util, result_queue))
    writer_thread.start()

    # Create threads for data generation
    chunk_size = ROWS // THREADS
    remaining_rows = ROWS
    batch = 0
    while remaining_rows > 0:
        active_threads = []
        
        for _ in range(min(THREADS, remaining_rows // BATCH_SIZE)):
            thread = Process(target=data_util.generate_random_data_chunk, args=(batch, BATCH_SIZE, result_queue))
            thread.start()
            active_threads.append(thread)
            remaining_rows -= BATCH_SIZE
            batch += 1
        
        # Wait for all active threads to finish
        for thread in active_threads:
            thread.join()
    
    # Send sentinel value to indicate end of data
    result_queue.put(None)
    
    # Wait for the database writer thread to finish
    writer_thread.join()

def summarize_table(db_path, table_name, data_util):
    """
    Summarizes the created table by printing the total number of rows, total number of possible combinations,
    and the frequency of each value for each column.
    
    Args:
    db_path (str): Path to the SQLite database file.
    table_name (str): Name of the table to summarize.
    possible_values (dict): A dictionary where keys are column names and values are lists of possible values for those columns.
    
    Returns:
    None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Calculate total number of rows
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]
    
    # Calculate total number of possible combinations
    total_combinations = data_util.generate_total_number_of_combinations()
    
    # Calculate frequency of each value for each column
    frequencies = {col: {value: 0 for value in data_util.possible_values[col]} for col in data_util.possible_values}
    for col in data_util.possible_values:
        cursor.execute(f"SELECT {col}, COUNT(*) FROM {table_name} GROUP BY {col}")
        for value, count in cursor.fetchall():
            frequencies[col][value] = count
    
    conn.close()
    
    # Print the summary
    print(f"Total rows: {total_rows} / {total_combinations} {(total_rows/total_combinations)*100:.2f}%)")
    for col, freq_dict in frequencies.items():
        print(f"\nColumn: {col}")
        for value, count in freq_dict.items():
            print(f"  {value}: {count}")

# Example usage
if __name__ == "__main__":
    config_path = 'config.json'  # Path to JSON config file
    data_util = DataUtil(config_path)

    db_path = data_util.db_name # Path to DB
    table_name = "data"

    create_database(db_path, table_name, data_util)
    print(f"Database '{db_path}' with table '{table_name}' has been created and populated.")
    summarize_table(db_path, table_name, data_util)
