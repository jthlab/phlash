import joblib
from platformdirs import user_cache_dir

memory = joblib.Memory(location=user_cache_dir("phlash"))
