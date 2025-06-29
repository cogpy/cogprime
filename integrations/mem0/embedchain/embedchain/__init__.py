import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from embedchain.app import App  # noqa: F401
from embedchain.client import Client  # noqa: F401
from embedchain.pipeline import Pipeline  # noqa: F401

# Explicit initialization function for setting up the user directory
def initialize_embedchain():
    """Initialize the embedchain module by setting up the user directory."""
    Client.setup()
