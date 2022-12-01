import logging

from sarjana.loggers.handlers import stream_handler
from sarjana.preamble import ExecutionOptions

datalogger = logging.getLogger(__name__)
datalogger.addHandler(stream_handler)

if ExecutionOptions.Mode == "debug":
    datalogger.setLevel(logging.DEBUG)
