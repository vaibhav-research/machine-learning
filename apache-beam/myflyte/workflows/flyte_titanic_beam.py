# flyte_titanic_beam.py

from flytekit import  workflow, ImageSpec
from flyte_beam_plugin.decorator import beam_task
from flyte_beam_plugin.beam_config import BeamJobConfig
from beam_titanic import run_titanic_pipeline
from typing import Optional


# image_spec = ImageSpec(
#     name="beam-titanic-image",
#     base_image="ghcr.io/flyteorg/flytekit:py3.11-1.10.2",
#     packages=["apache-beam[gcp]", "pandas"],
#     registry="localhost:30000"
# )

# @task(container_image=image_spec)
# def run_beam() -> str:
#     input_csv = "titanic.csv"
#     output_path = "titanic_metrics"
#     run_titanic_pipeline(input_csv,output_path)
#     return output_path

# @workflow
# def titanic_beam_pipeline() -> str:
#     return run_beam()

# if __name__ == "__main__":
#     titanic_beam_pipeline()

@beam_task(task_config=BeamJobConfig(runner="DirectRunner"))
def titanic_pipeline_task(input_csv: str, output_path: str) -> Optional[str]:
    print("Running pipeline with", input_csv, output_path)
    run_titanic_pipeline(input_csv=input_csv, output_path=output_path)
    print("Pipeline completed")
    return output_path

@workflow
def titanic_beam_pipeline() -> Optional[str]:
    return titanic_pipeline_task(input_csv="titanic.csv", output_path="titanic_metrics")

if __name__ == "__main__":
    print("Output path:", titanic_beam_pipeline())

