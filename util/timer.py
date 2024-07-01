import time
from contextlib import ContextDecorator

class Timer(ContextDecorator):
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *exc):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f'Elapsed time: {self.elapsed_time:.4f} seconds')

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.name:
            print(f'[{self.name}] Elapsed time: {self.elapsed_time:.3f}s')
        else:
            print(f'Elapsed time: {self.elapsed_time:.3f}s')
        return self.elapsed_time
