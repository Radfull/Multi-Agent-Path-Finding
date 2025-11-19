"""
Скрипт для анализа производительности и поиска реальных узких мест
"""
import cProfile
import pstats
from io import StringIO
from src.envs import eval_env1

def profile_run():
    """Профилирование одного запуска"""
    pr = cProfile.Profile()
    pr.enable()
    
    # Запускаем один тест
    result = eval_env1(search_type='a_star', used_dist='manh', seed=1, density_percent=50, w=1.0)
    
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Топ 30 функций
    
    print("=" * 80)
    print("ТОП 30 функций по времени выполнения (cumulative):")
    print("=" * 80)
    print(s.getvalue())
    
    # Также покажем по собственному времени
    s2 = StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(30)
    
    print("=" * 80)
    print("ТОП 30 функций по собственному времени (tottime):")
    print("=" * 80)
    print(s2.getvalue())

if __name__ == "__main__":
    profile_run()

