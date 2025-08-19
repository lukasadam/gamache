"""Configuration for the project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from dotenv import load_dotenv
from platformdirs import user_cache_dir
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

# Load .env if present (for tokens, URIs, etc.)
load_dotenv()


def _project_root() -> Path:
    """Get the project root directory."""
    # repo root = two levels up from this file (src/pkg/config.py)
    return Path(__file__).resolve().parents[2]


class Project(BaseModel):
    """Project metadata."""

    name: str
    seed: int = 42


class Paths(BaseModel):
    """Paths for various directories in the project."""

    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    figures_dir: Path
    models_dir: Path
    cache_dir: Optional[Path] = None  # default later

    @field_validator("*", mode="before")
    @classmethod
    def make_path(cls, v):
        """Convert string paths to Path objects.

        Parameters
        ----------
        v : str | Path
            The input path to convert.

        Returns
        -------
        Path
            The converted Path object.
        """
        # Convert strings to Paths
        return Path(v) if isinstance(v, str) else v

    @field_validator("cache_dir", mode="after")
    @classmethod
    def default_cache(cls, v):
        """Set default cache directory if not provided.

        Parameters
        ----------
        v : Optional[Path]
            User-provided cache directory path.

        Returns
        -------
        Path
            Default cache directory path if not provided, otherwise the user-provided path.
        """
        return v or Path(user_cache_dir("gamache"))

    def resolve_all(self, root: Path) -> "Paths":
        """Resolve all paths to be absolute relative to the project root.

        Parameters
        ----------
        root : Path
            The project root directory.

        Returns
        -------
        Paths
            A new Paths instance with all paths resolved to be absolute.
        """
        # Make all paths absolute relative to project root
        resolved = {}
        for k, p in self.model_dump().items():
            if p is None:
                resolved[k] = None
            else:
                rp = p if Path(p).is_absolute() else root / p
                resolved[k] = rp
        return Paths(**resolved)

    def ensure_dirs(self) -> None:
        """Ensure all directories exist, creating them if necessary."""
        for p in self.model_dump().values():
            if isinstance(p, Path):
                p.mkdir(parents=True, exist_ok=True)


class DatasetCfg(BaseModel):
    """Configuration for a dataset."""

    uri: str
    format: Literal["csv", "parquet", "json", "auto"] = "auto"
    storage_options: Dict[str, str] = {}


class Settings(BaseSettings):
    """Main configuration class for the project."""

    project: Project
    paths: Paths
    datasets: Dict[str, DatasetCfg] = {}

    class Config:
        """Pydantic configuration."""

        env_prefix = "gamache_"
        # Example: AWESOME_PYTHON_SERVICE_LR=0.01 overrides training.lr


def deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge two dictionaries.

    Parameters
    ----------
    a : dict
        The first dictionary.
    b : dict
        The second dictionary.

    Returns
    -------
    dict
        A new dictionary that is the result of merging `a` and `b`.
    """
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : Path
        Path to the YAML file.

    Returns
    -------
    dict
        The contents of the YAML file as a dictionary. Returns an empty dict if the file does not exist or is empty.
    """
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_settings(profile: Optional[str] = None) -> Settings:
    """Merge configs in order: base.yml -> local.yml (if present) -> {profile}.yaml (if given).

    Then resolve paths, ensure directories, and return typed Settings.

    Parameters
    ----------
    profile : Optional[str], optional
        Optional profile name to load additional settings from a specific YAML file (default is None).

    Returns
    -------
    Settings
        The merged and validated settings object.
    """
    root = _project_root()
    cfg_dir = root / "configs"

    base = load_yaml(cfg_dir / "base.yml")
    local = load_yaml(cfg_dir / "local.yml")

    merged = deep_merge(base, local)
    st = Settings.model_validate(merged)

    # Resolve & create directories
    st.paths = st.paths.resolve_all(root)
    st.paths.ensure_dirs()
    return st
