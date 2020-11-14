import logging
import sys


FORMATTER = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - " "%(funcName)s:%(lineno)d - %(message)s"
                    )
# Logger
def get_console_header():
    cheader = logging.StreamHandler(sys.stdout)
    cheader.setFormatter(FORMATTER)
    return cheader