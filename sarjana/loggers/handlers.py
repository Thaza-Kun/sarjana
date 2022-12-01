import logging


formatter = logging.Formatter("%(levelname)s::%(name)s::%(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
