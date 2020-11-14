import logging

from config.logging_conf import *

VERSION_PATH = "VERSION"

# loggin
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_header())
logger.propagate = False

with open(VERSION_PATH, 'r') as version:
    __version__ = version.read().strip()