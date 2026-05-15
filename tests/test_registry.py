"""Tests for the run registry — the cross-package contract."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kt_masterlog.registry import (
    SCHEMA_VERSION,
    register_run,
    registry_dir,
    update_run_status,
)


class TestRegistryDir:
    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KT_MASTERLOG_REGISTRY_DIR", str(tmp_path / "custom"))
        assert registry_dir() == tmp_path / "custom"

    def test_default_is_under_home(self, monkeypatch):
        monkeypatch.delenv("KT_MASTERLOG_REGISTRY_DIR", raising=False)
        assert registry_dir() == Path.home() / ".kt-masterlog" / "runs"


class TestRegisterRun:
    def test_writes_manifest_with_expected_fields(self, tmp_path):
        # isolated_registry autouse fixture already redirects to a temp dir
        run_id = register_run("smoke_proj", tmp_path / "log.csv")
        assert run_id is not None

        manifest_path = registry_dir() / f"{run_id}.json"
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["run_id"] == run_id
        assert data["project_name"] == "smoke_proj"
        assert data["csv_path"] == str((tmp_path / "log.csv").resolve())
        assert data["pid"] == os.getpid()
        assert data["status"] == "running"
        assert data["ended_at"] is None
        assert isinstance(data["hostname"], str)
        assert data["started_at"].endswith("+00:00")

    def test_each_call_produces_unique_run_id(self, tmp_path):
        ids = {register_run("p", tmp_path / "a.csv") for _ in range(5)}
        assert None not in ids
        assert len(ids) == 5

    def test_returns_none_and_does_not_raise_on_failure(
        self, tmp_path, monkeypatch
    ):
        """If the registry dir can't be created, we log + return None."""
        # Point at a path that can't be created (existing regular file).
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        monkeypatch.setenv("KT_MASTERLOG_REGISTRY_DIR", str(blocker / "child"))
        assert register_run("p", tmp_path / "log.csv") is None


class TestUpdateRunStatus:
    def test_updates_existing_manifest(self, tmp_path):
        run_id = register_run("p", tmp_path / "log.csv")
        update_run_status(run_id, "completed")

        data = json.loads((registry_dir() / f"{run_id}.json").read_text())
        assert data["status"] == "completed"
        assert data["ended_at"] is not None

    def test_failed_status(self, tmp_path):
        run_id = register_run("p", tmp_path / "log.csv")
        update_run_status(run_id, "failed")

        data = json.loads((registry_dir() / f"{run_id}.json").read_text())
        assert data["status"] == "failed"

    def test_none_run_id_is_noop(self):
        update_run_status(None, "completed")  # must not raise

    def test_missing_manifest_is_noop(self):
        update_run_status("nonexistent_run_id", "completed")  # must not raise


class TestOptimizeIntegration:
    """The optimize() orchestrator should register runs by default and
    transition status to completed (or failed on exception)."""

    def test_optimize_registers_and_completes(
        self, tmp_path, tiny_data, builder
    ):
        from kt_masterlog import TunerConfig, optimize

        x_train, y_train, x_val, y_val = tiny_data
        config = TunerConfig(
            project_name="registry_smoke",
            output_dir=str(tmp_path),
            strategy="random",
            max_trials=1,
            search_epochs=1,
            early_stop_patience=10,
        )
        optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )

        manifests = list(registry_dir().glob("*.json"))
        assert len(manifests) == 1
        data = json.loads(manifests[0].read_text())
        assert data["project_name"] == "registry_smoke"
        assert data["status"] == "completed"
        assert data["ended_at"] is not None
        assert data["csv_path"].endswith("registry_smoke_master_log.csv")

    def test_register_run_false_skips_registry(
        self, tmp_path, tiny_data, builder
    ):
        from kt_masterlog import TunerConfig, optimize

        x_train, y_train, x_val, y_val = tiny_data
        config = TunerConfig(
            project_name="no_registry",
            output_dir=str(tmp_path),
            strategy="random",
            max_trials=1,
            search_epochs=1,
            early_stop_patience=10,
            register_run=False,
        )
        optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )

        # Registry dir may not even exist; if it does, it must be empty.
        if registry_dir().exists():
            assert list(registry_dir().glob("*.json")) == []
