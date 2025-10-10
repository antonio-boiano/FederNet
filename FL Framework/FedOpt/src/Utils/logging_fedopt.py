import logging
import sys

def get_logging_level(level_str):
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    return levels.get(level_str, logging.DEBUG)

def setup_logger(name, level_str="debug"):
    logger = logging.getLogger(name)
    logger.setLevel(get_logging_level(level_str))
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s"))
    logger.addHandler(stdout)
    logger.propagate = False
    return logger
