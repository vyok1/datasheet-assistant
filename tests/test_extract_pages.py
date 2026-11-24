"""Tests for extract_pages.py CLI paths. No PDFs or network access are required."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_pages.py"


@pytest.fixture
def extractor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("extract_pages", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(module, "DEFAULT_CONFIG", tmp_path / "configs" / "corpus.yaml")
    monkeypatch.setattr(module, "DEFAULT_OUTPUT", tmp_path / "data" / "pages" / "extracted.json")
    module.DEFAULT_CONFIG.parent.mkdir(parents=True)
    module.DEFAULT_CONFIG.write_text(
        yaml.safe_dump({"documents": [{"id": "missing_doc", "title": "Original", "pages": [2]}]}),
        encoding="utf-8",
    )
    return module


def test_extract_pages_defaults_preserve_original_paths(extractor):
    assert extractor.main([]) == 0
    data = json.loads(extractor.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert data["documents"][0]["id"] == "missing_doc"
    assert data["documents"][0]["title"] == "Original"
    assert data["documents"][0]["requested_pages"] == [2]


def test_extract_pages_custom_config_and_output_do_not_overwrite_baseline(
    extractor,
    tmp_path: Path,
):
    custom_config = tmp_path / "configs" / "corpus_full.yaml"
    custom_config.write_text(
        yaml.safe_dump({"documents": [{"id": "full_doc", "title": "Full", "pages": None}]}),
        encoding="utf-8",
    )
    custom_output = tmp_path / "data" / "pages" / "extracted_full.json"
    baseline = extractor.DEFAULT_OUTPUT
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("baseline fixture", encoding="utf-8")

    assert extractor.main(["--config", str(custom_config), "--output", str(custom_output)]) == 0

    result = json.loads(custom_output.read_text(encoding="utf-8"))
    assert result["documents"][0]["id"] == "full_doc"
    assert result["documents"][0]["title"] == "Full"
    assert result["documents"][0]["requested_pages"] is None
    assert baseline.read_text(encoding="utf-8") == "baseline fixture"
