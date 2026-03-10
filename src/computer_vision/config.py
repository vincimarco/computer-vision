import pathlib

import dvc.api

# ---------------------------------------------------------------------------- #
#                                     PATHS                                    #
# ---------------------------------------------------------------------------- #

RAW_DATA_DIR = pathlib.Path("data/1.raw/ECDUY")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

INTERIM_DATA_DIR = pathlib.Path("data/2.interim/ECDUY")
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DATA_DIR = pathlib.Path("data/3.final/ECDUY")
FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------- #
#                                  PARAMS.YAML                                 #
# ---------------------------------------------------------------------------- #
params = dvc.api.params_show()
