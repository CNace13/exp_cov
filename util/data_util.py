import json
import random
import math

class DataUtil:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.db_name = self.config['db_name']
        self.columns = list(self.config['columns'].keys())
        self.possible_values = {col: self.config['columns'][col]['possible_values'] for col in self.columns}
        self.distributions = {col: self.config['columns'][col]['distributions'] for col in self.columns}
        self.widths = {col: math.ceil(math.log2(len(self.config['columns'][col]['possible_values']))) for col in self.columns}

    def generate_total_number_of_combinations(self):
        # Calculate total number of possible combinations
        total_combinations = 1
        for values in self.possible_values.values():
            total_combinations *= len(values)
        return total_combinations

    def generate_random_data(self, rows):
        data = []
        for _ in range(rows):
            row = []
            pk_id = 0
            cumulative_widths = 0
            for col in self.columns:
                value = random.choices(self.possible_values[col], weights=self.distributions[col], k=1)[0]
                row.append(value)
                pk_id |= value << cumulative_widths
                cumulative_widths += self.widths[col]
            print(pk_id, row)
            data.append((pk_id, row))
        return data
    
    # For multithreaded data generation
    def generate_random_data_chunk(self, thread, rows, result_queue):
        data = []
        print(f"Thread: {thread} Rows: {rows} BEGIN")
        for _ in range(rows):
            row = []
            pk_id = 0
            cumulative_widths = 0
            for col in self.columns:
                value = random.choices(self.possible_values[col], weights=self.distributions[col], k=1)[0]
                row.append(value)
                pk_id |= value << cumulative_widths
                cumulative_widths += self.widths[col]
            print(pk_id, row)
            data.append((pk_id, row))
        print(f"Thread: {thread} DONE")
        result_queue.put(data)

    def generate_all_combinations(self):
        from itertools import product
        all_combinations = list(product(*[self.possible_values[col] for col in self.columns]))
        hashed_combinations = []
        for combination in all_combinations:
            pk_id = 0
            cumulative_widths = 0
            for value, col in zip(combination, self.columns):
                pk_id |= value << cumulative_widths
                cumulative_widths += self.widths[col]
            hashed_combinations.append((combination, pk_id))
        return hashed_combinations

# Example usage
if __name__ == "__main__":
    config_path = 'config.json'  # Path to your JSON config file
    data_generator = DataGenerator(config_path)

    # Generate random data
    random_data = data_generator.generate_random_data(1000)  # Generate 1000 rows of random data
    for row, pk_id in random_data:
        print(f"Row: {row}, Hash: {pk_id}")

    # Generate all possible combinations
    all_combinations = data_generator.generate_all_combinations()
    for combination, pk_id in all_combinations:
        print(f"Combination: {combination}, Hash: {pk_id}")
