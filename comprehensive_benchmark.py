"""
Комплексный тест производительности
"""
from timeit import default_timer as timer
from src.envs import eval_env1, eval_env2
import statistics

def benchmark_env(env_func, env_name, iterations=10):
    """Запускает бенчмарк для одного окружения"""
    print(f"\n{'='*60}")
    print(f"Testing {env_name}")
    print(f"{'='*60}")
    
    times = []
    for i in range(iterations):
        start = timer()
        result = env_func(search_type='a_star', used_dist='manh', seed=1, density_percent=50, w=1.0)
        elapsed = timer() - start
        times.append(elapsed)
        if i < 3:  # Show first 3 runs
            print(f"  Run {i+1}: {elapsed:.4f}s, steps: {len(result)}")
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults after {iterations} runs:")
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Std dev: {std_time:.4f}s")
    print(f"  Min:     {min_time:.4f}s")
    print(f"  Max:     {max_time:.4f}s")
    
    return avg_time, std_time, min_time, max_time

if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE BENCHMARK TEST")
    print("=" * 60)
    
    # Test env1
    avg1, std1, min1, max1 = benchmark_env(eval_env1, "env1", iterations=10)
    
    # Test env2
    avg2, std2, min2, max2 = benchmark_env(eval_env2, "env2", iterations=10)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"env1 average: {avg1:.4f}s")
    print(f"env2 average: {avg2:.4f}s")
    print(f"Overall average: {(avg1 + avg2) / 2:.4f}s")
    print(f"{'='*60}")


