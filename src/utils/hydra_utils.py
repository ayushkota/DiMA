from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path


def setup_config(config_path):
    # Reset Hydra to avoid conflicts if already initialized
    GlobalHydra.instance().clear()

    # Convert config path to an absolute path relative to the project root
    project_root = Path(__file__).resolve().parents[2]
    config_dir = (project_root / config_path).resolve()

    # Initialize Hydra with the absolute config directory
    initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None
    )

    # Load the configuration
    cfg = compose(config_name="config")

    return cfg
