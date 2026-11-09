import json

import pytest

from cashflow_hedge.cli import main


VALID = {"curve": {"knots": [2, 5, 10], "rates": [.025, .032, .038]}, "cashflows": {"times": [6], "amounts": [100]}, "knot_shifts_bp": [100, 0, -100]}


def test_cli_roundtrip_and_input_readonly(tmp_path, capsys):
    path = tmp_path / "input.json"
    original = json.dumps(VALID).encode()
    path.write_bytes(original)
    assert main([str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["present_value"] > 80
    assert report["stress"]["full_repricing_change"] > 0
    assert len(report["dollar_dv01"]) == 3
    assert path.read_bytes() == original


@pytest.mark.parametrize("input_text", ["[]", "{}", '{"curve":{},"curve":{}}', "invalid", " "*(1024*1024+1), json.dumps({**VALID, "unexpected": 1}), json.dumps({**VALID, "cashflow_shifts_bp": None}), json.dumps({**VALID, "knot_shifts_bp": None})], ids=["array", "empty", "duplicate", "syntax", "oversize", "unknown", "both-fields", "null-shift"])
def test_cli_invalid_inputs(tmp_path, capsys, input_text):
    path = tmp_path / "input.json"
    path.write_text(input_text)
    assert main([str(path)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("cashflow-risk:")


def test_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "missing.json")]) == 1
    assert "cashflow-risk:" in capsys.readouterr().err
