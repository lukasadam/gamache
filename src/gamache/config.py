from __future__ import annotations
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from platformdirs import user_cache_dir
from dotenv import load_dotenv

# Load .env if present (for tokens, URIs, etc.)
load_dotenv()


def _project_root() -> Path:
    # repo root = two levels up from this file (src/pkg/config.py)
    return Path(__file__).resolve().parents[2]


class Project(BaseModel):
    name: str
    seed: int = 42


class Paths(BaseModel):
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
        # Convert strings to Paths
        return Path(v) if isinstance(v, str) else v

    @field_validator("cache_dir", mode="after")
    @classmethod
    def default_cache(cls, v):
        return v or Path(user_cache_dir("gamache"))

    def resolve_all(self, root: Path) -> "Paths":
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
        for p in self.model_dump().values():
            if isinstance(p, Path):
                p.mkdir(parents=True, exist_ok=True)


class DatasetCfg(BaseModel):
    uri: str
    format: Literal["csv", "parquet", "json", "auto"] = "auto"
    storage_options: Dict[str, str] = {}


class Training(BaseModel):
    batch_size: int = 32
    epochs: int = 1
    lr: float = 1e-3


class Settings(BaseSettings):
    project: Project
    paths: Paths
    datasets: Dict[str, DatasetCfg] = {}
    training: Training = Training()

    class Config:
        env_prefix = "gamache_"
        # Example: AWESOME_PYTHON_SERVICE_LR=0.01 overrides training.lr


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_settings(profile: Optional[str] = None) -> Settings:
    """
    Merge configs in order: base.yml -> local.yml (if present) -> {profile}.yaml (if given).
    Then resolve paths, ensure directories, and return typed Settings.
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
