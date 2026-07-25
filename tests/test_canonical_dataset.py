"""Tests for eval.canonical_dataset and training_track_gate."""

import json
from pathlib import Path

import pytest
import yaml

from eval.canonical_dataset import (
    CANONICAL_TRAINING_DATASET_PATH,
    assert_recipe_uses_canonical_dataset,
    canonical_sft_sha256,
    load_canonical,
    sft_sha256_from_canonical_text,
)
from eval.training_track_gate import (
    gate_training_pr,
    is_training_track_pr,
    validate_changed_paths,
    validate_pr_body_canonical_pin,
    validate_pr_body_proof_bundle,
    validate_recipe_paths_in_ref,
)

_VALID_PROOF_BUNDLE_URL = "https://huggingface.co/gittensor-model-hub/sparkdistill-2026-07-11-qwen3.5-4b-mining-001"


def _training_pr_body(*, proof_bundle_url: str | None = _VALID_PROOF_BUNDLE_URL) -> str:
    pin = load_canonical()
    proof_line = proof_bundle_url or "pending"
    return (
        "- [x] **Training/evaluation improvement**\n"
        f"- Canonical dataset URL: {pin['hf_url']}\n"
        f"- Pinned sft_sha256: `{pin['mix_manifest']['sft_sha256']}`\n"
        f"- Proof-bundle URL: {proof_line}\n"
    )


def test_load_canonical_pin():
    pin = load_canonical(Path("datasets/canonical.json"))
    assert pin["repo_id"] == "gittensor-model-hub/sparkproof-mining"
    assert pin["mix_manifest"]["sft_sha256"]


def test_recipe_rejects_non_canonical_paths():
    recipe = {
        "datasets": [{"path": "data/processed/triton_sft.jsonl"}],
    }
    issues = assert_recipe_uses_canonical_dataset(recipe)
    assert any(CANONICAL_TRAINING_DATASET_PATH in issue for issue in issues)


def test_training_track_checkbox():
    assert is_training_track_pr("- [x] **Training/evaluation improvement**")
    assert is_training_track_pr("- [x] Training/evaluation improvement")
    assert not is_training_track_pr("- [x] **Dataset track submission**")
    assert not is_training_track_pr("- [x] Dataset track submission")


def test_forbidden_training_paths():
    issues = validate_changed_paths(["eval/gen_triton_kernels.py"])
    assert any("forbidden pattern" in issue for issue in issues)
    issues = validate_changed_paths(["scripts/prepare_triton_kernels.sh"])
    assert issues
    assert validate_changed_paths(["datasets/canonical.json"]) == []


def test_validate_pr_body_requires_canonical_citation():
    pin = load_canonical()
    body = (
        f"Dataset URL: {pin['hf_url']}\n"
        f"sha `{pin['mix_manifest']['sft_sha256']}`\n"
        f"Proof-bundle URL: {_VALID_PROOF_BUNDLE_URL}\n"
    )
    assert validate_pr_body_canonical_pin(body) == []


def test_validate_pr_body_rejects_missing_proof_bundle():
    pin = load_canonical()
    body = (
        f"Dataset URL: {pin['hf_url']}\n"
        f"sha `{pin['mix_manifest']['sft_sha256']}`\n"
        "Proof-bundle URL: pending after local train + eval\n"
    )
    issues = validate_pr_body_proof_bundle(body)
    assert issues
    assert any("pending" in issue.lower() or "published" in issue.lower() for issue in issues)


def test_gate_training_pr_rejects_missing_proof_bundle(tmp_path: Path):
    report = gate_training_pr(
        head_ref="HEAD",
        changed_paths=["recipes/qwen3.5-4b-phase1/sft-mining.yaml"],
        pr_body=_training_pr_body(proof_bundle_url="pending after local train + eval"),
        verify_hf_pin=False,
        verify_proof_bundle=False,
    )
    assert report["label"] == "training:REJECT"
    assert any("Proof-bundle" in issue for issue in report["issues"])


def test_gate_training_pr_rejects_local_generator(tmp_path: Path):
    recipe = tmp_path / "recipes/qwen3.5-4b-phase1/sft-triton.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [{"path": "data/processed/triton_sft.jsonl"}],
            }
        ),
        encoding="utf-8",
    )
    report = gate_training_pr(
        head_ref="HEAD",
        changed_paths=["eval/gen_triton_kernels.py", recipe.as_posix()],
        pr_body=_training_pr_body(),
        verify_hf_pin=False,
        verify_proof_bundle=False,
    )
    assert report["label"] == "training:REJECT"
    assert not report["verified"]
    assert report["issues"]


def test_validate_recipe_paths_in_worktree(tmp_path: Path, monkeypatch):
    recipe = tmp_path / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump({"datasets": [{"path": CANONICAL_TRAINING_DATASET_PATH}]}),
        encoding="utf-8",
    )

    def _fake_show(ref, path):
        if path.endswith("recipes/demo/sft.yaml"):
            return recipe.read_text(encoding="utf-8")
        return None

    monkeypatch.setattr("eval.training_track_gate._git_show", _fake_show)
    assert validate_recipe_paths_in_ref("HEAD", ["recipes/demo/sft.yaml"]) == []


# datasets/canonical.json is miner-controlled on a training-track PR (_ALLOWED_ALWAYS),
# and _canonical_sft_sha256s_for_pr_window parses every revision in the PR's pin-grace
# window through sft_sha256_from_canonical_text. Every unusable shape must be None.
_UNUSABLE_CANONICAL_TEXTS = [
    ("invalid_json", "{not json"),
    ("payload_not_an_object", json.dumps([1, 2])),
    ("mix_manifest_absent", json.dumps({"repo_id": "x"})),
    ("mix_manifest_null", json.dumps({"mix_manifest": None})),
    ("mix_manifest_list", json.dumps({"mix_manifest": [{"sft_sha256": "a" * 64}]})),
    ("mix_manifest_string", json.dumps({"mix_manifest": "a" * 64})),
    ("mix_manifest_number", json.dumps({"mix_manifest": 5})),
    ("digest_too_short", json.dumps({"mix_manifest": {"sft_sha256": "abc"}})),
]


@pytest.mark.parametrize("case, text", _UNUSABLE_CANONICAL_TEXTS, ids=[c for c, _ in _UNUSABLE_CANONICAL_TEXTS])
def test_sft_sha256_from_canonical_text_returns_none_for_unusable_shapes(case, text):
    assert sft_sha256_from_canonical_text(text) is None


def test_sft_sha256_from_canonical_text_reads_a_valid_pin():
    assert sft_sha256_from_canonical_text(json.dumps({"mix_manifest": {"sft_sha256": "a" * 64}})) == "a" * 64


def test_canonical_sft_sha256_raises_valueerror_on_non_object_mix_manifest(tmp_path):
    """Callers catch ValueError from this function; AttributeError escaped them."""
    pin = tmp_path / "canonical.json"
    pin.write_text(json.dumps({"mix_manifest": ["not", "an", "object"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="mix_manifest must be a JSON object"):
        canonical_sft_sha256(pin)


def test_canonical_sft_sha256_still_raises_on_a_bad_digest(tmp_path):
    pin = tmp_path / "canonical.json"
    pin.write_text(json.dumps({"mix_manifest": {"sft_sha256": "abc"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="64-char hex digest"):
        canonical_sft_sha256(pin)


def test_gate_training_pr_survives_a_non_object_mix_manifest_at_the_pr_head(monkeypatch):
    """The gate must reject the PR, not abort the Training track gate job."""
    import eval.training_track_gate as gate

    monkeypatch.setattr(
        gate,
        "_git_show",
        lambda ref, path: (
            json.dumps({"mix_manifest": [{"sft_sha256": "a" * 64}]}) if path.endswith("canonical.json") else ""
        ),
    )

    report = gate_training_pr(
        head_ref="HEAD",
        changed_paths=["recipes/foo.yaml", "datasets/canonical.json"],
        pr_body="- [x] **Training/evaluation improvement**",
        merge_base_ref="origin/main",
        verify_hf_pin=False,
        verify_proof_bundle=False,
    )
    assert report["verified"] is False
    assert report["label"] == "training:REJECT"
    assert report["issues"]
