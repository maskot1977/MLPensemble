from logging import DEBUG, basicConfig, getLogger

logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
