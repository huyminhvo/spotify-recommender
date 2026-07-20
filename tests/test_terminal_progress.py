from utils import terminal_progress


def test_terminal_progress_throttles_and_reports_completion(monkeypatch, capsys):
    times = iter([0.0, 10.0, 20.0])
    monkeypatch.setattr(terminal_progress.time, "monotonic", lambda: next(times))
    progress = terminal_progress.TerminalProgress("test", updates=10)

    progress(1, 100, "playlist", 0, "first")
    assert capsys.readouterr().out == ""

    progress(10, 100, "playlist", 0, "strategy")
    output = capsys.readouterr().out
    assert "[test] 10/100 ( 10.0%)" in output
    assert "split=1 strategy=strategy" in output
    assert "eta=1m30s" in output

    progress(100, 100, "playlist", 1, "done")
    output = capsys.readouterr().out
    assert "[test] 100/100 (100.0%)" in output
    assert "eta=0s" in output
