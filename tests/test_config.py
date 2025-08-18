from pathlib import Path

import yaml

import gamache.config as config


def test_deep_merge_merges_dicts():
    a = {"a": 1, "b": {"x": 1, "y": 2}}
    b = {"b": {"y": 99, "z": 3}, "c": 42}
    merged = config.deep_merge(a, b)
    assert merged == {
        "a": 1,
        "b": {"x": 1, "y": 99, "z": 3},
        "c": 42,
    }


def test_load_yaml_reads_and_returns_dict(tmp_path: Path):
    yaml_path = tmp_path / "test.yml"
    yaml_path.write_text("a: 1\nb: foo\n")
    out = config.load_yaml(yaml_path)
    assert out == {"a": 1, "b": "foo"}

    # Non-existent file should return {}
    assert config.load_yaml(tmp_path / "missing.yml") == {}


def test_load_settings_merges_and_creates_dirs(tmp_path: Path, monkeypatch):
    # Create fake project root with configs/
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()

    base_cfg = {
        "project": {"name": "demo"},
        "paths": {
            "data_dir": "data",
            "raw_dir": "raw",
            "interim_dir": "interim",
            "processed_dir": "processed",
            "figures_dir": "figures",
            "models_dir": "models",
        },
    }
    local_cfg = {
        "project": {"seed": 123},
        "training": {"epochs": 10},
    }

    (cfg_dir / "base.yml").write_text(yaml.safe_dump(base_cfg))
    (cfg_dir / "local.yml").write_text(yaml.safe_dump(local_cfg))

    # Patch _project_root to point to tmp_path
    monkeypatch.setattr(config, "_project_root", lambda: tmp_path)

    st = config.load_settings()

    # Project values merged correctly
    assert st.project.name == "demo"
    assert st.project.seed == 123  # overridden by local.yml
    assert st.training.epochs == 10

    # Paths resolved relative to tmp_path
    assert st.paths.data_dir == tmp_path / "data"
    assert st.paths.models_dir == tmp_path / "models"

    # Directories actually created
    assert st.paths.data_dir.exists()
    assert st.paths.models_dir.exists()
