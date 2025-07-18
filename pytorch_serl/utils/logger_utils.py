import logging
import os
import sys


def get_logger(out_dir):
    logger = logging.getLogger("Exp")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(process)d %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def setup_logging(out_dir, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(process)d %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "run.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
