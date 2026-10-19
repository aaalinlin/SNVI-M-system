
import os

from segcore.utilities.default_n_proc_DA import get_allowed_n_proc_DA

default_num_processes = 8 if 'SEGCORE_EXPORT_WORKERS' not in os.environ else int(os.environ['SEGCORE_EXPORT_WORKERS'])

ANISO_THRESHOLD = 3  # determines when a sample is considered anisotropic (3 means that the spacing in the low
# resolution axis must be 3x as large as the next largest spacing)

default_n_proc_DA = get_allowed_n_proc_DA()
