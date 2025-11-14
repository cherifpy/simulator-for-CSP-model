import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Simulate task scheduling.")
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level (default: INFO)."
        )
        self.parser.add_argument(
            "--config",
            help="Specify the path of the config file"
        )
        self.parser.add_argument(
            "--plot-gantt",
            action="store_true",
            help="Plot the gantt chart of the simulation"
        )
        self.parser.add_argument(
            "--homogeneous",
            action="store_true",
            help="Choose the homogeneous version of the simulator"
        )

    def parse(self):
        return self.parser.parse_args()
