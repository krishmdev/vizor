import os

__version__ = "0.2.0"

# Models load only from the project-local cache fetched by `make models` (see vizor/models.py).
# HF_HOME has to be set before huggingface_hub is imported anywhere.
if "HF_HOME" not in os.environ:
    from vizor.models import project_root

    os.environ["HF_HOME"] = str(project_root() / ".models")
# Keep CPU use modest by default (VIZOR_THREADS overrides). On macOS the index uses NumPy rather
# than faiss, because faiss-cpu, torch and scikit-learn each bundle a libomp and loading several
# in one process crashes (see retrieve/index.py).
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("VIZOR_THREADS", "2"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
