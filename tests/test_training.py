r"""Tests for the run names and the training configurations of heimdall.tools."""

from pathlib import Path

from heimdall.tools import generate_run_name, load_configuration


def test_generate_run_name_same_id() -> None:
    r"""Determines if the jobs of a same training share their name, unlike two trainings."""

    assert generate_run_name("ENC", run_id="1_0") == generate_run_name("ENC", run_id="1_0")
    assert generate_run_name("ENC", run_id="1_0") != generate_run_name("ENC", run_id="2_0")


def test_load_configuration_base(tmp_path: Path) -> None:
    r"""Determines if a file inherits its common file, its own values replacing the common ones."""

    (tmp_path / "common.yml").write_text(
        'State:\n  "saving" : [true]\nTraining:\n  "a" : [1]\n  "b" : [2]\n'
    )
    (tmp_path / "file.yml").write_text(
        'Base : "common.yml"\nTraining:\n  "b" : [3]\nModel:\n  "c" : [4]\n'
    )

    config = load_configuration(tmp_path / "file.yml")

    assert len(config) == 1
    assert config[0] == {
        "State": {"saving": True},
        "Training": {"a": 1, "b": 3},
        "Model": {"c": 4},
    }


def test_load_configuration_encoders() -> None:
    r"""Determines if every encoder has a complete configuration, with whole validation batches."""

    folder = Path(__file__).parents[1] / "experiments" / "encoders" / "configs" / "encoder"
    paths = sorted(folder.glob("encoder-*.yml"))

    assert len(paths) == 7
    for path in paths:
        config = load_configuration(path)[0]
        training = config["Training"]

        assert "Base" not in config
        assert "batches_validation" not in training
        assert config["State"] == {
            "saving": True,
            "resume_optimizer": True,
            "checkpoint_name": None,
        }
        assert training["samples_validation"] % training["batch_size_per_gpu"] == 0
