from __future__ import annotations

import pytest

from vsession.cli import main


def test_cli_rejects_math_only_method_for_gsm8k_without_traceback(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["dry-run", "--dataset", "gsm8k", "--method", "no_goal"])

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "only available for MATH500" in error
    assert "Traceback" not in error


def test_cli_reports_incompatible_sampling_arguments_without_traceback(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["dry-run", "--dataset", "math500", "--method", "direct", "--do-sample"])

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "sampling requires temperature > 0" in error
    assert "Traceback" not in error
