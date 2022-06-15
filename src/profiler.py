"""Profile 관련 함수 모음

사용 방법:
    * command line > snake_viz profile_data

"""
import io
import pstats
from cProfile import Profile


def profile_run(command: str, path: str) -> None:
    """
    profile the command

    Args:
        command: profile target
        path: where the profile will be saved (e.g., '../profile_data/prof_data.prof')
    Returns:
        No return
    """
    profiler = Profile()
    profiler.run(command)
    str_io = io.StringIO()
    stats = pstats.Stats(profiler, stream=str_io).sort_stats('ncalls')
    stats.print_stats()

    # stats 파일 저장
    stats.dump_stats(path)
