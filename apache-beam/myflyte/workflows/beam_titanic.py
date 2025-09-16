# beam_titanic.py

import apache_beam as beam
import csv
from typing import Dict, List, Optional
import logging

def parse_csv(line: str) -> List[Dict]:  # Return a list, not a generator
    try:
        value = list(csv.reader([line]))[0]
        record = {
            "Survived": int(value[1]),
            "Sex": value[4],
            "Age": float(value[5]) if value[5] else None
        }
        if (
            record["Age"] is not None
            and record["Sex"] in ("male", "female")
            and record["Survived"] in (0, 1)
        ):
            return [record]
    except Exception as e:
        logging.debug(f"Skipping invalid line: {line}. Reason: {e}")
    return []

def is_valid(record: Dict) -> bool:
    return (
        record.get("Age") is not None
        and record.get("Sex") in ("male", "female")
        and record.get("Survived") in (0, 1)
    )

def run_titanic_pipeline(input_csv: str, output_path: str):
    with beam.Pipeline() as p:
        records = (
            p
            | "Read CSV" >> beam.io.ReadFromText(input_csv, skip_header_lines=1)
            | "Parse CSV" >> beam.FlatMap(parse_csv)
            | "Filter Invalid" >> beam.Filter(is_valid)
        )

        avg_age_survivor = (
            records
            | "Filter Survivors" >> beam.Filter(lambda r: r["Survived"] == 1)
            | "Extract Age" >> beam.Map(lambda r: r["Age"])
            | "Avg Age" >> beam.CombineGlobally(beam.combiners.MeanCombineFn())
        )

        male_survivors = (
            records
            | "Filter Males" >> beam.Filter(lambda r: r["Sex"] == "male")
            | "Male Survival" >> beam.Map(lambda r: r["Survived"])
            | "Avg Male Survival Rate" >> beam.CombineGlobally(beam.combiners.MeanCombineFn())
        )

        female_survivors = (
            records
            | "Filter Females" >> beam.Filter(lambda r: r["Sex"] == "female")
            | "Female Survival" >> beam.Map(lambda r: r["Survived"])
            | "Avg Female Survival Rate" >> beam.CombineGlobally(beam.combiners.MeanCombineFn())
        )

        total_passenger = (
            records
            | "Count Passengers" >> beam.combiners.Count.Globally()
        )

        # Writing results
        result = (
            [
                avg_age_survivor | "Wrap Avg Age Survivors" >> beam.Map(lambda v: ("avg_age_survivors", v)),
                female_survivors | "Wrap Female Survival" >> beam.Map(lambda v: ("survival_rate_female", v)),
                male_survivors | "Wrap Male Survival" >> beam.Map(lambda v: ("survival_rate_male", v)),
                total_passenger | "Wrap Total Passengers" >> beam.Map(lambda v: ("total_passengers", v)),
            ]
            | "Merge Results" >> beam.Flatten()
            | "Group All" >> beam.GroupByKey()
            | "Format Output" >> beam.Map(lambda kv: f"{kv[0]}: {list(kv[1])[0]}")  # Get single value from iterable
            | "WriteResult" >> beam.io.WriteToText(output_path)
        )
    return result
