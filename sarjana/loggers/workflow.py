import logging

from sarjana.loggers.handlers import stream_handler
from sarjana.preamble import ExecutionOptions

flowlogger = logging.getLogger(__name__)
flowlogger.addHandler(stream_handler)

if ExecutionOptions.Mode == "debug":
    flowlogger.setLevel(logging.DEBUG)
