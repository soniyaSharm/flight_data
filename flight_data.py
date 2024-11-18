import json
import random
import os
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import concurrent.futures
import time
from typing import Dict, List, Tuple


class FlightDataProcessor:
    def __init__(self):
        self.base_dir = Path("tmp/flights")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.num_files = 5000
        self.num_cities = random.randint(100, 200)
        self.cities = [f"CITY_{i}" for i in range(self.num_cities)]
        self.null_prob = random.uniform(0.001, 0.005)

    def generate_flight_record(self) -> dict:
        """Generate a single flight record with possibility of NULL values"""
        origin = random.choice(self.cities)
        dest = random.choice([c for c in self.cities if c != origin])

        record = {
            "date": (datetime.now() + timedelta(days=random.randint(-30, 30))).strftime(
                "%Y-%m-%d"
            ),
            "origin_city": origin,
            "destination_city": dest,
            "flight_duration_secs": random.randint(1800, 36000),
            "passengers": random.randint(50, 400),
        }

        for key in record:
            if random.random() < self.null_prob:
                record[key] = None

        return record

    def generate_file(self, file_idx: int) -> None:
        """Generate a single JSON file with random flight records"""
        try:
            num_records = random.randint(50, 100)
            records = []

            for _ in range(num_records):
                record = self.generate_flight_record()

                if (
                    record["origin_city"] not in self.cities
                    or record["destination_city"] not in self.cities
                ):
                    continue

                records.append(record)

            now = datetime.now()
            month_year = now.strftime("%m-%Y")
            origin_city = random.choice(self.cities)

            dir_path = Path(self.base_dir) / month_year
            dir_path.mkdir(parents=True, exist_ok=True)

            file_path = dir_path / f"{month_year}-{origin_city}-{file_idx}-flights.json"

            with open(file_path, "w") as f:
                json.dump(records, f)

        except Exception as e:
            print(f"Error generating file {file_idx}: {str(e)}")

    def generate_data(self) -> None:
        """Generate all flight data files using parallel processing"""
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(self.generate_file, range(self.num_files)))
        print(f"Generating files in: {self.base_dir}")

    def is_dirty_record(self, record: dict) -> bool:
        """Check if a record contains any NULL values or is missing required keys"""
        required_keys = [
            "origin_city",
            "destination_city",
            "flight_duration_secs",
            "passengers",
        ]
        for key in required_keys:
            if key not in record or record[key] is None:
                return True
        return False

    def analyze_data(self) -> Tuple[dict, float]:
        """Analyze the generated flight data"""
        start_time = time.time()

        total_records = 0
        dirty_records = 0
        flight_durations = {}
        city_passengers = {city: {"arrived": 0, "left": 0} for city in self.cities}

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue

                try:
                    with open(os.path.join(root, file), "r") as f:
                        try:
                            records = json.load(f)
                        except json.JSONDecodeError as e:
                            print(f"Error reading file {file}: {str(e)}")
                            continue

                    for record in records:
                        total_records += 1

                        if self.is_dirty_record(record):
                            dirty_records += 1
                            continue

                        if "passengers" not in record or record["passengers"] is None:
                            print(
                                f"Skipping record in {file}: Missing or None 'passengers'"
                            )
                            dirty_records += 1
                            continue

                        dest = record.get("destination_city")
                        if dest not in flight_durations:
                            flight_durations[dest] = []

                        if (
                            "flight_duration_secs" in record
                            and record["flight_duration_secs"] is not None
                        ):
                            flight_durations[dest].append(
                                record["flight_duration_secs"]
                            )

                        passengers = record["passengers"]
                        city_passengers[record["destination_city"]][
                            "arrived"
                        ] += passengers
                        city_passengers[record["origin_city"]]["left"] += passengers

                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue

        stats = {
            "total_records": total_records,
            "dirty_records": dirty_records,
            "duration_stats": self._calculate_duration_stats(flight_durations),
            "passenger_stats": self._calculate_passenger_stats(city_passengers),
        }

        run_duration = time.time() - start_time
        return stats, run_duration

    def _calculate_duration_stats(self, flight_durations: Dict[str, List[int]]) -> dict:
        """Calculate AVG and P95 for top 25 destination cities"""
        stats = {}
        for city, durations in flight_durations.items():
            if len(durations) > 0:
                stats[city] = {
                    "avg": float(np.mean(durations)),
                    "p95": float(np.percentile(durations, 95)),
                }

        return dict(
            sorted(
                stats.items(), key=lambda x: len(flight_durations[x[0]]), reverse=True
            )[:25]
        )

    def _calculate_passenger_stats(
        self, city_passengers: Dict[str, Dict[str, int]]
    ) -> dict:
        """Find cities with max passengers arrived and left"""
        max_arrived = max(city_passengers.items(), key=lambda x: x[1]["arrived"])
        max_left = max(city_passengers.items(), key=lambda x: x[1]["left"])

        return {
            "max_arrived": {
                "city": max_arrived[0],
                "passengers": max_arrived[1]["arrived"],
            },
            "max_left": {"city": max_left[0], "passengers": max_left[1]["left"]},
        }


def main():
    processor = FlightDataProcessor()

    print("Starting data generation...")
    processor.generate_data()
    print("Data generation completed.")

    print("\nStarting data analysis...")
    stats, run_duration = processor.analyze_data()

    print(f"\nAnalysis Results:")
    print(f"Total records processed: {stats['total_records']:,}")
    print(f"Dirty records found: {stats['dirty_records']:,}")
    print(f"Total run duration: {run_duration:.2f} seconds")

    print("\nTop 25 Destination Cities Flight Duration Stats:")
    for city, duration_stats in stats["duration_stats"].items():
        print(f"{city}:")
        print(f"  AVG: {duration_stats['avg']:.2f} seconds")
        print(f"  P95: {duration_stats['p95']:.2f} seconds")

    print("\nPassenger Statistics:")
    print(f"City with most arrivals: {stats['passenger_stats']['max_arrived']['city']}")
    print(
        f"Total passengers arrived: {stats['passenger_stats']['max_arrived']['passengers']:,}"
    )
    print(f"City with most departures: {stats['passenger_stats']['max_left']['city']}")
    print(
        f"Total passengers departed: {stats['passenger_stats']['max_left']['passengers']:,}"
    )


if __name__ == "__main__":
    main()
