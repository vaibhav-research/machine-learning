# task.py
from __future__ import annotations
import inspect
from typing import Any, Dict, Optional

import apache_beam as beam
from flytekit import PythonFunctionTask
from flytekit.loggers import logger
from flytekit.core.context_manager import FlyteContext
from flyte_beam_plugin.beam_config import BeamJobConfig


class BeamFunctionTask(PythonFunctionTask[BeamJobConfig]):
    """
    Flyte task wrapper for Apache Beam pipelines.

    Usage:
        @beam_task(task_config=BeamJobConfig(...))
        def my_pipeline(pipeline: Optional[beam.Pipeline], input_path: str) -> str:
            # If `pipeline` is present, build transforms on it.
            # Otherwise, create your own with beam.Pipeline() and return a value.
            ...
    """
    # ----- stable task type -----
    _TASK_TYPE = "beam"

    def __init__(self, task_function, task_config: Optional[BeamJobConfig] = None, **kwargs):
        # Avoid duplicate passing
        kwargs.pop("task_config", None)
        super().__init__(task_function=task_function, task_config=task_config or BeamJobConfig(), **kwargs)
        self._user_fn = task_function

    # ----- expose a stable type string for registration -----
    @property
    def task_type(self) -> str:
        return self._TASK_TYPE

    # ----- surface runner/config to control plane -----
    def get_custom(self, _ctx: Optional[FlyteContext] = None) -> Dict[str, Any]:
        # what the control-plane sees (serialized into the task template)
        c = self.task_config
        return {
            "runner": c.runner,
            "project": c.project,
            "region": c.region,
            "temp_location": c.temp_location,
        }

    # (Optional) helpful for debugging / programmatic access
    def get_config(self, _ctx: Optional[FlyteContext] = None) -> BeamJobConfig:
        return self.task_config

    # ----- LOCAL EXECUTION PATH -----
    def local_execute(self, ctx: FlyteContext, **kwargs):
        """
        Local path used by `pyflyte run` / local workflow runs.
        - Creates Beam PipelineOptions from config
        - If user function expects `pipeline` (by name or annotation), inject a managed Pipeline
        - Else, call user function as-is
        - Return the user function's return value verbatim
        """
        logger.info("Launching Beam task (local) with config: %s", self.get_custom())

        options = self._make_pipeline_options()

        # Should we inject a managed pipeline?
        wants_pipeline = _user_fn_wants_pipeline(self._user_fn)

        if wants_pipeline:
            with beam.Pipeline(options=options) as pipeline:
                # Inject pipeline keyword if present
                bound_kwargs = _bind_pipeline_arg(self._user_fn, kwargs, pipeline)
                result = self._user_fn(**bound_kwargs)
        else:
            # User manages Beam pipeline themselves (fine for DirectRunner)
            result = self._user_fn(**kwargs)

        return result

    # In a follow-up PR you’ll add a non-local path that executes in a container on a platform runner.
    # def execute(self, ctx: FlyteContext, **kwargs):
    #     ...

    def _make_pipeline_options(self) -> beam.pipeline.PipelineOptions:
        c = self.task_config
        opts: Dict[str, Any] = {"runner": c.runner}
        if c.temp_location:
            opts["temp_location"] = c.temp_location
        if c.project:
            opts["project"] = c.project
        if c.region:
            opts["region"] = c.region
        return beam.pipeline.PipelineOptions(flags=[], **opts)


# ---------- helpers ----------

def _user_fn_wants_pipeline(fn) -> bool:
    """Return True if the user function has a parameter named 'pipeline'
    or annotated as apache_beam.Pipeline."""
    sig = inspect.signature(fn)
    for name, p in sig.parameters.items():
        if name == "pipeline":
            return True
        ann = p.annotation
        try:
            # tolerate forward refs / typing issues
            if ann is beam.Pipeline:
                return True
        except Exception:
            pass
    return False

def _bind_pipeline_arg(fn, kwargs: Dict[str, Any], pipeline: beam.Pipeline) -> Dict[str, Any]:
    """Return kwargs ensuring the user fn receives the managed 'pipeline' kwarg if requested."""
    sig = inspect.signature(fn)
    if "pipeline" in sig.parameters:
        if "pipeline" in kwargs:
            # user passed their own; prefer managed pipeline & warn
            logger.warning("Overriding user-provided 'pipeline' with managed pipeline.")
        new_kwargs = dict(kwargs)
        new_kwargs["pipeline"] = pipeline
        return new_kwargs
    return kwargs
