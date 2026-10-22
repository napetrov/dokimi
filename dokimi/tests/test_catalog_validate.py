# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Spec and mirror review of ``catalog scan`` (``dokimi.external.validate``), and its ``--format json`` output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dokimi.cli import build_parser
from dokimi.external import read_skill_manifest, review_skills
from dokimi.utils import parse_frontmatter, parse_frontmatter_strict


def _skill(root: Path, rel: str, name: str | None = None, description: str = "Use for things.",
           extra: str = "", body: str = "# Body\n") -> Path:
    """Write ``<root>/<rel>/SKILL.md``; ``name`` None leaves the key out of the frontmatter."""
    skill_dir = root / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    head = f"name: {name}\n" if name is not None else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\n{head}description: {description}\n{extra}---\n{body}", encoding="utf-8")
    return skill_dir


def _scan_json(
    root: Path, capsys, *, kind: str = "skills", strict: bool = False
) -> tuple[dict, int]:
    """Run ``catalog scan --format json``; return the parsed report and the exit code (0 when none raised)."""
    argv = ["catalog", "scan", "--root", str(root), "--kind", kind, "--format", "json"]
    args = build_parser().parse_args(argv + (["--strict"] if strict else []))
    code = 0
    try:
        args.func(args)
    except SystemExit as exc:
        code = exc.code
    return json.loads(capsys.readouterr().out), code


def _reviews_by_name(root: Path) -> dict:
    entries = read_skill_manifest(root)
    return {e.name: r for e, r in zip(entries, review_skills(entries), strict=True)}


def test_name_that_is_not_the_directory_name_is_an_error(tmp_path):
    _skill(tmp_path / "skills", "writing-rules", name="Writing Hookify Rules")

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "error"
    codes = {f.code for f in review.findings}
    assert {"name-mismatch", "name-format"} <= codes


def test_title_case_name_matching_directory_is_still_not_kebab_case(tmp_path):
    _skill(tmp_path / "skills", "Writing", name="Writing")

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "error"
    assert [f.code for f in review.findings] == ["name-format"]


def test_kebab_case_name_matching_directory_is_ok(tmp_path):
    _skill(tmp_path / "skills", "linux-perf-2", name="linux-perf-2")

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "ok"
    assert review.findings == []


def test_name_missing_from_frontmatter_is_a_warning(tmp_path):
    _skill(tmp_path / "skills", "no-name", name=None)

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "warn"
    assert [f.code for f in review.findings] == ["name-missing"]


def test_description_length_limit_is_1024_characters(tmp_path):
    # "Use for " keeps a when-clause in the description, so only the length is in question.
    _skill(tmp_path / "skills", "at-limit", name="at-limit", description="Use for " + "d" * 1016)
    _skill(
        tmp_path / "skills", "over-limit", name="over-limit", description="Use for " + "d" * 1017
    )

    reviews = _reviews_by_name(tmp_path)
    assert reviews["at-limit"].state == "ok"
    over = reviews["over-limit"]
    assert over.state == "error"
    assert [f.code for f in over.findings] == ["description-too-long"]


def test_identical_skill_md_is_a_warning_naming_the_first_copy(tmp_path):
    # A plugin copy with an extra bundled folder is still the same SKILL.md: the model sees the same text.
    _skill(tmp_path / "plugins", "x", name="x", description="Use for x.")
    copy = _skill(tmp_path / "skills", "x", name="x", description="Use for x.")
    (copy / "agents").mkdir()
    (copy / "agents" / "openai.yaml").write_text("a: 1\n", encoding="utf-8")

    reviews = review_skills(read_skill_manifest(tmp_path))
    originals = [r for r in reviews if r.duplicate_of is None]
    copies = [r for r in reviews if r.duplicate_of is not None]
    assert len(originals) == 1 and len(copies) == 1
    assert copies[0].duplicate_of == "plugins/x"
    assert copies[0].state == "warn"
    assert copies[0].findings[-1].code == "duplicate"
    assert copies[0].content_sha256 == originals[0].content_sha256


def test_different_skill_md_with_the_same_name_is_not_a_duplicate(tmp_path):
    _skill(tmp_path / "plugins", "x", name="x", description="Use for x.")
    _skill(tmp_path / "skills", "x", name="x", description="Use for x, a different text.")

    reviews = review_skills(read_skill_manifest(tmp_path))
    assert all(r.duplicate_of is None for r in reviews)
    # Not a duplicate, but a mirror that has drifted: both copies say so.
    assert [[f.code for f in r.findings] for r in reviews] == [["mirror-drift"], ["mirror-drift"]]


def test_disable_model_invocation_is_reported_as_a_flag_not_a_finding(tmp_path):
    _skill(tmp_path / "skills", "hidden", name="hidden",
           extra="disable-model-invocation: true\n")
    _skill(tmp_path / "skills", "shown", name="shown")

    reviews = _reviews_by_name(tmp_path)
    assert reviews["hidden"].disable_model_invocation is True
    assert reviews["hidden"].state == "ok"
    assert reviews["shown"].disable_model_invocation is False


def test_json_report_carries_the_findings_and_exits_0_on_a_spec_error_without_strict(
    tmp_path, capsys
):
    _skill(tmp_path / "skills", "claude-api", name="Claude API")
    report, code = _scan_json(tmp_path, capsys)

    assert code == 0
    [row] = report["skills"]
    assert row["state"] == "error"
    assert row["error"] is None
    assert {f["code"] for f in row["findings"]} >= {"name-mismatch", "name-format"}
    assert row["content_sha256"] and len(row["content_sha256"]) == 64


def test_strict_makes_a_spec_error_exit_1(tmp_path, capsys):
    _skill(tmp_path / "skills", "claude-api", name="Claude API")
    report, code = _scan_json(tmp_path, capsys, strict=True)

    assert code == 1
    assert report["skills"][0]["state"] == "error"


def test_strict_does_not_fail_on_warnings(tmp_path, capsys):
    _skill(tmp_path / "plugins", "x", name="x")
    _skill(tmp_path / "skills", "x", name="x")
    _skill(tmp_path / "skills", "no-name", name=None)
    _, code = _scan_json(tmp_path, capsys, strict=True)

    assert code == 0


def test_json_report_on_warnings_only_exits_0(tmp_path, capsys):
    _skill(tmp_path / "plugins", "x", name="x")
    _skill(tmp_path / "skills", "x", name="x")
    _skill(tmp_path / "skills", "hidden", name="hidden", extra="disable-model-invocation: yes\n")
    report, code = _scan_json(tmp_path, capsys)

    assert code == 0
    states = {row["path"].split(str(tmp_path) + "/")[1]: row["state"] for row in report["skills"]}
    assert states == {"plugins/x": "ok", "skills/x": "warn", "skills/hidden": "ok"}
    hidden = next(r for r in report["skills"] if r["name"] == "hidden")
    assert hidden["disable_model_invocation"] is True
    assert report["tasks"] == []


def test_json_report_keeps_load_errors_as_error_rows(tmp_path, capsys):
    (tmp_path / "skills.yaml").write_text(
        'skills:\n  - name: ghost\n    catalog_path: skills/does-not-exist\n', encoding="utf-8")
    report, code = _scan_json(tmp_path, capsys)

    assert code == 1
    [row] = report["skills"]
    assert row["state"] == "error"
    assert "does-not-exist" in row["error"]


def test_text_output_lists_each_finding_under_its_skill(tmp_path, capsys):
    _skill(tmp_path / "skills", "claude-api", name="Claude API")
    args = build_parser().parse_args(["catalog", "scan", "--root", str(tmp_path), "--kind", "skills"])
    args.func(args)  # no SystemExit: a spec error is printed, and only --strict makes it fail
    out = capsys.readouterr().out
    assert "# skills (1 declared" in out and "1 error" in out
    assert "ERROR: name 'Claude API' is not the directory name" in out
    assert "  error name-mismatch:" in out
    assert "  error name-format:" in out


BARE_COLON = "Use when: the user asks for a Claude API call"


def test_frontmatter_with_a_bare_colon_is_a_frontmatter_parse_error(tmp_path):
    # ``description: Use when: ...`` is not valid YAML; the lenient parser reads it as no metadata.
    _skill(tmp_path / "skills", "claude-api", name="claude-api", description=BARE_COLON)

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "error"
    [finding] = review.findings
    assert finding.code == "frontmatter-parse"
    assert finding.level == "error"
    assert finding.message.startswith(
        "frontmatter is not valid YAML: mapping values are not allowed"
    )
    assert "\n" not in finding.message


def test_broken_yaml_is_reported_in_place_of_the_missing_description_error(tmp_path, capsys):
    _skill(tmp_path / "skills", "claude-api", name="claude-api", description=BARE_COLON)
    report, code = _scan_json(tmp_path, capsys)

    assert code == 1  # a skill that does not load exits 1 with or without --strict
    [row] = report["skills"]
    assert row["state"] == "error"
    assert "missing a 'description'" in row["error"]
    assert [f["code"] for f in row["findings"]] == ["frontmatter-parse"]
    assert row["content_sha256"] is None and row["duplicate_of"] is None


def test_strict_parse_raises_and_the_lenient_parse_still_returns_no_metadata():
    text = f"---\nname: claude-api\ndescription: {BARE_COLON}\n---\nbody\n"

    assert parse_frontmatter(text) == ({}, "body")
    with pytest.raises(yaml.YAMLError):
        parse_frontmatter_strict(text)


def _codes(review) -> list[str]:
    return [f.code for f in review.findings]


def test_relative_link_to_a_file_that_is_not_there_is_an_error(tmp_path):
    _skill(tmp_path / "skills", "guided", name="guided", description="Use for guided work.",
           body="See [the guide](references/guide.md).\n")
    bundled = _skill(
        tmp_path / "skills",
        "bundled",
        name="bundled",
        description="Use for bundled work.",
        body="Details in [the guide](references/guide.md).\n",
    )
    (bundled / "references").mkdir()
    (bundled / "references" / "guide.md").write_text("# G\n", encoding="utf-8")

    reviews = _reviews_by_name(tmp_path)
    assert reviews["guided"].state == "error"
    assert _codes(reviews["guided"]) == ["link-missing"]
    assert reviews["bundled"].state == "ok"


def test_link_that_exists_from_the_catalog_root_says_the_base_is_wrong(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print(1)\n", encoding="utf-8")
    _skill(tmp_path / "skills", "runner", name="runner", body="Run [it](scripts/run.py).\n")

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "error"
    [finding] = review.findings
    assert finding.code == "link-missing"
    assert "the base is wrong" in finding.message


def test_links_in_code_fences_urls_and_anchors_are_not_checked(tmp_path):
    body = (
        "Read [the site](https://example.com/docs) or [below](#usage).\n"
        "```markdown\n"
        "See [an example](references/missing.md).\n"
        "```\n"
        "~~~\n"
        "[also](scripts/missing.sh)\n"
        "~~~\n"
    )
    _skill(tmp_path / "skills", "clean", name="clean", body=body)

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "ok"
    assert review.findings == []


def test_skill_dir_variable_resolves_inside_the_skill(tmp_path):
    skill = _skill(tmp_path / "skills", "vars", name="vars",
                   body="[a](${CLAUDE_SKILL_DIR}/references/a.md) [b](${CLAUDE_SKILL_DIR}/b.md)\n")
    (skill / "references").mkdir()
    (skill / "references" / "a.md").write_text("# A\n", encoding="utf-8")

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert _codes(review) == ["link-missing"]
    assert "'${CLAUDE_SKILL_DIR}/b.md'" in review.findings[0].message


def test_link_out_of_the_skill_dir_to_an_existing_file_is_a_warning(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    _skill(tmp_path / "skills", "reach", name="reach", body="See [notes](../../shared/notes.md).\n")

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "warn"
    assert _codes(review) == ["link-escapes-skill"]


def test_link_out_of_the_skill_dir_to_a_missing_file_is_an_error(tmp_path):
    _skill(tmp_path / "skills", "reach", name="reach", body="See [notes](../../nowhere.md).\n")

    [review] = review_skills(read_skill_manifest(tmp_path))
    assert review.state == "error"
    assert _codes(review) == ["link-missing"]


def test_dotdot_code_span_to_an_existing_file_outside_the_skill_is_a_warning(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    _skill(
        tmp_path / "skills",
        "reach",
        name="reach",
        body="Keep `../../shared/notes.md` next to it. Edit `docs/missing.md` in your project.\n",
    )

    [review] = review_skills(read_skill_manifest(tmp_path))
    # The code span naming a missing project path is not checked: only the escape is reported.
    assert review.state == "warn"
    assert _codes(review) == ["path-escapes-skill"]


def test_description_without_a_when_clause_is_a_warning(tmp_path):
    _skill(
        tmp_path / "skills", "plain", name="plain", description="Builds release reports from logs."
    )
    _skill(tmp_path / "skills", "said", name="said",
           description="Summarises deploy outcomes. Use when a release is cut.")

    reviews = _reviews_by_name(tmp_path)
    assert reviews["plain"].state == "warn"
    assert _codes(reviews["plain"]) == ["description-no-when"]
    assert reviews["said"].state == "ok"


def test_hidden_skill_is_exempt_from_the_description_checks(tmp_path):
    _skill(tmp_path / "skills", "hidden", name="hidden", description="Builds release reports.",
           extra="disable-model-invocation: true\n")
    _skill(tmp_path / "skills", "twin", name="twin", description="Builds release reports.",
           extra="disable-model-invocation: true\n")

    reviews = _reviews_by_name(tmp_path)
    assert reviews["hidden"].state == "ok"
    assert reviews["twin"].state == "ok"


def test_near_duplicate_descriptions_warn_on_both_skills(tmp_path):
    _skill(tmp_path / "skills", "style-check", name="style-check",
           description="Use when reviewing Python pull requests for style and bugs.")
    _skill(tmp_path / "skills", "style-lint", name="style-lint",
           description="Use when reviewing Python pull requests for style and typing.")

    reviews = _reviews_by_name(tmp_path)
    assert _codes(reviews["style-check"]) == ["description-near-duplicate"]
    assert _codes(reviews["style-lint"]) == ["description-near-duplicate"]
    assert "skills/style-lint" in reviews["style-check"].findings[0].message


def test_unrelated_descriptions_are_not_near_duplicates(tmp_path):
    _skill(tmp_path / "skills", "deploy", name="deploy",
           description="Use when deploying containers to Kubernetes clusters.")
    _skill(tmp_path / "skills", "notes", name="notes",
           description="Use when writing release notes for a changelog.")

    reviews = review_skills(read_skill_manifest(tmp_path))
    assert all(r.state == "ok" for r in reviews)


def test_drifted_copies_of_one_skill_warn_on_both_and_name_each_other(tmp_path):
    _skill(tmp_path / "plugins", "x", name="x", description="Use for x.", body="# Old\n")
    _skill(tmp_path / "skills", "x", name="x", description="Use for x.", body="# New\n")

    reviews = review_skills(read_skill_manifest(tmp_path))
    assert all(r.duplicate_of is None for r in reviews)
    assert [_codes(r) for r in reviews] == [["mirror-drift"], ["mirror-drift"]]
    assert "differs from plugins/x" in reviews[1].findings[0].message
    assert "differs from skills/x" in reviews[0].findings[0].message


def test_identical_copies_are_a_duplicate_and_not_drift(tmp_path):
    _skill(tmp_path / "plugins", "x", name="x", description="Use for x.")
    _skill(tmp_path / "skills", "x", name="x", description="Use for x.")

    reviews = review_skills(read_skill_manifest(tmp_path))
    assert [_codes(r) for r in reviews if r.duplicate_of is not None] == [["duplicate"]]
    assert all("mirror-drift" not in _codes(r) for r in reviews)


def test_new_warnings_do_not_fail_the_scan_but_a_missing_link_does_under_strict(tmp_path, capsys):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    _skill(tmp_path / "skills", "reach", name="reach", description="Builds reports.",
           body="See [notes](../../shared/notes.md).\n")
    report, code = _scan_json(tmp_path, capsys, strict=True)
    assert code == 0
    assert report["skills"][0]["state"] == "warn"
    assert {f["code"] for f in report["skills"][0]["findings"]} == {
        "description-no-when", "link-escapes-skill"}

    _skill(tmp_path / "skills", "broken", name="broken", body="See [gone](gone.md).\n")
    report, code = _scan_json(tmp_path, capsys, strict=True)
    assert code == 1
    states = {row["name"]: row["state"] for row in report["skills"]}
    assert states == {"reach": "warn", "broken": "error"}
