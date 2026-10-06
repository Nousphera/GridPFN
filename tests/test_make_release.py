"""Synthetic packaging tests; no real release, data or application files touched."""

import hashlib
import json
import zipfile

import pytest

from scripts import make_release as release


@pytest.mark.parametrize(
    "name",
    [
        "wiki/private.md",
        "docs/WIKI/private.md",
        "results/report.json",
        ".git/config",
        "config/.ENV",
        "config/.env.example",
        ".VENV-model/settings.json",
        "dataset/home.csv",
        "dataset/home.CSV",
        "dataset/home.npz",
        "model.CKPT",
        "model.SAFETENSORS",
        "model.PKL",
        "signing.pem",
        "credentials.json",
        "site/evidence.json",
        "site/results.svg",
        "site/foundation-evidence.json",
        "site/foundation.html",
        "gridpfn/unreviewed.py",
    ],
)
def test_unreviewed_private_data_keys_and_old_assets_fail_closed(tmp_path, name):
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Synthetic path probe, not private content.")
    with pytest.raises(ValueError):
        release.reviewed_payload(tmp_path, [name])


@pytest.mark.parametrize("name", ["../README.md", "/README.md", "./README.md", "a//b", "a\\b", ""])
def test_noncanonical_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid"):
        release.reviewed_payload(tmp_path, [name])


def test_exclusions_cannot_be_bypassed_by_case_or_allowlist_extension(tmp_path, monkeypatch):
    names = ["dataset/private.NPZ", "docs/WIKI/private.md", "model.PT"]
    monkeypatch.setattr(release, "RELEASE_PATHS", frozenset(names))
    for name in names:
        with pytest.raises(ValueError, match="Private|Raw data"):
            release.reviewed_payload(tmp_path, [name])


@pytest.mark.parametrize("parent_link", [False, True])
def test_even_allowlisted_symlinks_are_rejected(tmp_path, parent_link):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    if parent_link:
        (outside / "index.html").write_text("Synthetic external file")
        (root / "energy_assistant").mkdir()
        (root / "energy_assistant/static").symlink_to(outside, target_is_directory=True)
        name = "energy_assistant/static/index.html"
    else:
        (outside / "source.md").write_text("Synthetic external file")
        (root / "README.md").symlink_to(outside / "source.md")
        name = "README.md"
    with pytest.raises(ValueError, match="Symlinks"):
        release.reviewed_payload(root, [name])


def test_known_credential_scan_remains_secondary_to_path_review(tmp_path):
    (tmp_path / "README.md").write_text("tabpfn_" + "sk_" + "x" * 24)
    with pytest.raises(ValueError, match="known credential"):
        release.reviewed_payload(tmp_path, ["README.md"])


def test_missing_duplicate_and_nonregular_entries_are_not_omitted(tmp_path):
    with pytest.raises(ValueError, match="Missing regular"):
        release.reviewed_payload(tmp_path, ["README.md"])
    (tmp_path / "README.md").mkdir()
    with pytest.raises(ValueError, match="Missing regular"):
        release.reviewed_payload(tmp_path, ["README.md"])
    (tmp_path / "README.md").rmdir()
    (tmp_path / "README.md").write_text("Readme")
    with pytest.raises(ValueError, match="duplicate"):
        release.reviewed_payload(tmp_path, ["README.md", "README.md"])


def test_valid_payload_retains_compatibility_configs_and_current_app(tmp_path):
    names = [
        "README.md",
        "dataset/README.md",
        "configs/submission.json",
        "configs/foundation_prototype.json",
        "configs/gridpfn.toml",
        "configs/seasonal.json",
        "energy_assistant/static/explanation.js",
        "energy_assistant/static/index.html",
        "hems_assistant.py",
        "site/performance.json",
        "site/performance.svg",
    ]
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Synthetic reviewed file")
    assert set(release.reviewed_payload(tmp_path, names)) == set(names)


def setup_archive(tmp_path, monkeypatch, *, dirty=False, mutate_after_read=False):
    names = ["README.md", "LICENSE", "site/performance.json"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Apache License" if name == "LICENSE" else "Synthetic archive test")
    output = tmp_path / "release.zip"
    events = []

    def evidence(path):
        assert path == tmp_path / "site/performance.json"
        events.append("evidence")

    def git(command, **kwargs):
        assert events[0] == "evidence"
        events.append(command[1])
        if command[1] == "status":
            return " M README.md" if dirty else ""
        if command[1] == "ls-files":
            return ("\0".join(names) + "\0").encode()
        if command[1] == "rev-parse":
            if mutate_after_read:
                (tmp_path / "README.md").write_text("Changed after bytes were captured")
            return "a" * 40
        raise AssertionError(command)

    monkeypatch.setattr(release, "ROOT", tmp_path)
    # Evidence validity has separate exhaustive tests. Stub it ONLY to exercise
    # archive construction with synthetic files in this temporary directory.
    monkeypatch.setattr(release, "load_evidence", evidence)
    monkeypatch.setattr(release.subprocess, "check_output", git)
    monkeypatch.setattr("sys.argv", ["make_release", "--output", str(output)])
    return output, names, events


def test_archive_manifest_hashes_exact_captured_bytes(tmp_path, monkeypatch):
    output, names, events = setup_archive(tmp_path, monkeypatch, mutate_after_read=True)
    release.main()
    assert events == ["evidence", "status", "ls-files", "rev-parse"]
    with zipfile.ZipFile(output) as archive:
        report = json.loads(archive.read("gridpfn/RELEASE_MANIFEST.json"))
        assert set(report["files"]) == set(names)
        assert set(archive.namelist()) == {"gridpfn/" + n for n in names} | {
            "gridpfn/RELEASE_MANIFEST.json"
        }
        for name, digest in report["files"].items():
            assert hashlib.sha256(archive.read("gridpfn/" + name)).hexdigest() == digest
        assert archive.read("gridpfn/README.md") == b"Synthetic archive test"
        assert "not a guarantee" in report["content_review"]


def test_dirty_commit_gate_precedes_archive_creation(tmp_path, monkeypatch):
    output, _, events = setup_archive(tmp_path, monkeypatch, dirty=True)
    with pytest.raises(ValueError, match="Commit the reviewed release"):
        release.main()
    assert events == ["evidence", "status"]
    assert not output.exists()


def test_archive_output_symlink_and_source_overwrite_are_rejected(tmp_path, monkeypatch):
    output, _, _ = setup_archive(tmp_path, monkeypatch)
    other = tmp_path / "elsewhere"
    other.write_text("Preserve")
    output.symlink_to(other)
    with pytest.raises(ValueError, match="Symlinks"):
        release.main()
    assert other.read_text() == "Preserve"
    output.unlink()
    monkeypatch.setattr("sys.argv", ["make_release", "--output", str(tmp_path / "README.md")])
    with pytest.raises(ValueError, match="overwrite"):
        release.main()


def test_output_parent_traversal_cannot_overwrite_source(tmp_path, monkeypatch):
    setup_archive(tmp_path, monkeypatch)
    (tmp_path / "subdir").mkdir()
    monkeypatch.setattr(
        "sys.argv", ["make_release", "--output", str(tmp_path / "subdir/../README.md")]
    )
    with pytest.raises(ValueError, match="overwrite"):
        release.main()
