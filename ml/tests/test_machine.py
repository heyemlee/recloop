"""The cost runs' quiet-machine protocol (machine.measure_quietly), with docker and the load
average scripted."""

from ml.eval import machine


def _script(monkeypatch, containers, load1=1.0):
    """containers: what each `docker ps` call returns, in call order."""
    answers = iter(containers)
    monkeypatch.setattr(machine, "containers", lambda: next(answers))
    monkeypatch.setattr(machine.os, "getloadavg", lambda: (load1, load1, load1))


def _quietly(fn, **kw):
    return machine.measure_quietly(fn, max_wait=0, log=lambda _: None, **kw)


def test_one_attempt_on_a_quiet_machine(monkeypatch):
    # wait_quiet, the watch's first check, its last check, the end state
    _script(monkeypatch, [[], [], [], []])
    result, rec = _quietly(lambda: "done")
    assert result == "done" and len(rec["attempts"]) == 1
    assert rec["quiet_start"] and not rec["disturbed"]
    a = rec["attempts"][0]
    assert a["start"]["load1"] == 1.0 and a["start"]["containers"] == []
    assert a["end"]["containers"] == [] and a["containers_during"] == []


def test_runs_once_more_after_a_container_ran(monkeypatch):
    _script(monkeypatch, [[], [], ["aihd-test-pg"], ["aihd-test-pg"], [], [], [], []])
    calls = []
    result, rec = _quietly(lambda: calls.append(1) or len(calls))
    assert result == 2 and len(rec["attempts"]) == 2
    assert rec["attempts"][0]["containers_during"] == ["aihd-test-pg"]
    assert rec["attempts"][1]["containers_during"] == []
    assert rec["quiet_start"] and not rec["disturbed"]


def test_keeps_a_second_disturbed_attempt(monkeypatch):
    _script(monkeypatch, [["kafka"]] * 8, load1=3.0)
    calls = []
    _, rec = _quietly(lambda: calls.append(1))
    assert len(calls) == 2  # no third attempt
    assert rec["disturbed"] and not rec["quiet_start"]


def test_docker_not_answering_counts_as_no_containers(monkeypatch):
    _script(monkeypatch, [None, None, None, None])
    _, rec = _quietly(lambda: None)
    assert len(rec["attempts"]) == 1 and rec["quiet_start"] and not rec["disturbed"]
    assert rec["attempts"][0]["start"]["containers"] is None
