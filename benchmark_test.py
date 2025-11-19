"""
Тест производительности для сравнения оптимизированной и неоптимизированной версий
"""
from timeit import default_timer as timer
from src.envs import eval_env1
import statistics

def benchmark(iterations=5):
    """Запускает бенчмарк и возвращает среднее время"""
    times = []
    for i in range(iterations):
        start = timer()
        result = eval_env1(search_type='a_star', used_dist='manh', seed=1, density_percent=50, w=1.0)
        elapsed = timer() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s, steps: {len(result)}")
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    return avg_time, std_time, times

if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK TEST")
    print("=" * 60)
    print(f"Testing with 5 iterations...")
    print()
    
    avg, std, all_times = benchmark(5)
    
    print()
    print("=" * 60)
    print(f"Average time: {avg:.4f}s")
    print(f"Std deviation: {std:.4f}s")
    print(f"Min: {min(all_times):.4f}s")
    print(f"Max: {max(all_times):.4f}s")
    print("=" * 60)


