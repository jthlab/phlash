import logging
from logging import getLogger  # noqa: F401

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.getLogger("eastbay").setLevel(logging.DEBUG)
