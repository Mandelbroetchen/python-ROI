import time
import functools

def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print()
        print(f"[time] {func.__name__}: {execution_time:.6f} seconds")
        print(f"[time] {args}")
        print(f"[time] {kwargs}")
        return result
    return wrapper