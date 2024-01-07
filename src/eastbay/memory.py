import appdirs
from joblib import Memory

memory = Memory(appdirs.user_cache_dir(__name__.split(".")[0]), verbose=0)
