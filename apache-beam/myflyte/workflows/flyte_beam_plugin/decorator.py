# decorator.py
from typing import Optional, Callable
from flyte_beam_plugin.task import BeamFunctionTask
from flyte_beam_plugin.beam_config import BeamJobConfig

def beam_task(task_config: Optional[BeamJobConfig] = None, **task_kwargs):
    """
    Decorator to declare a Beam task.

    Example:
        @beam_task(BeamJobConfig(runner="DirectRunner"))
        def my_beam_fn(pipeline, input_path: str) -> str:
            ...
            return "ok"
    """
    def decorator(fn: Callable):
        return BeamFunctionTask(fn, task_config=task_config, **task_kwargs)
    return decorator
