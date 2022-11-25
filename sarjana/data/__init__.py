import logging

datalogger = logging.getLogger(f"{__name__}:data:")
datalogger.addHandler(stream_handler)