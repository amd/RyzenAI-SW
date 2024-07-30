##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import logging
import os

LINE_WIDTH = 80
LINE_SEPARATER = "-" * LINE_WIDTH


def get_cachedir():
    if os.environ.get("HF_HOME", None):
        return os.environ["HF_HOME"]
    else:
        return "./_cache"


class RyzenAILogger:
    def __init__(self, name):
        # Log directory
        self.logdir = "_logs"
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        # Log file
        self.logfile = self.logdir + "/" + name.replace("/", "_") + ".log"

    def start(self):
        # Start logging
        logging.basicConfig(
            filename=self.logfile,
            filemode="w",
            level=logging.CRITICAL,
        )

    def stop(self):
        # Stop logging
        logging.shutdown()
