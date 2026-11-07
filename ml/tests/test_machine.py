"""The cost runs' quiet-machine protocol (machine.measure_quietly), with docker and the load
average scripted."""

from ml.eval import machine


def _script(monkeypatch, containers, load1=1.0, swapouts=None):
    """containers: what each `docker ps` call returns, in call order; swapouts: vm_stat's
    swap-out count at each state() reading (constant if None)."""
    answers = iter(containers)
    counts = iter(swapouts) if swapouts is not None else None
    monkeypatch.setattr(machine, "containers", lambda: next(answers))
    monkeypatch.setattr(machine.os, "getloadavg", lambda: (load1, load1, load1))
    monkeypatch.setattr(machine, "memory_free_percent", lambda: 60)
    monkeypatch.setattr(
        machine,
        "vm_counters",
        lambda: {"page_size": 16384, "swapins": 0, "swapouts": next(counts) if counts else 7},
    )


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
    _script(
        monkeypatch,
        [[], [], ["aihd-test-pg"], ["aihd-test-pg"], [], [], [], []],
        swapouts=[100, 103, 103, 103],
    )
    calls = []
    result, rec = _quietly(lambda: calls.append(1) or len(calls))
    assert result == 2 and len(rec["attempts"]) == 2
    assert rec["attempts"][0]["containers_during"] == ["aihd-test-pg"]
    assert rec["attempts"][1]["containers_during"] == []
    assert rec["quiet_start"] and not rec["disturbed"]
    # the first attempt swapped out 3 pages, the kept one nothing
    assert rec["attempts"][0]["swapped_out_bytes"] == 3 * 16384
    assert rec["swapped_out_bytes"] == 0
    assert rec["attempts"][1]["start"]["memory_free_percent"] == 60


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


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    10952.
Swapins:                                    41572123.
Swapouts:                                   48485911.
"""


def test_vm_counters_parses_vm_stat(monkeypatch):
    monkeypatch.setattr(machine, "_run", lambda cmd, timeout=30: VM_STAT)
    vm = machine.vm_counters()
    assert vm == {"page_size": 16384, "swapins": 41572123, "swapouts": 48485911}
    later = dict(vm, swapouts=vm["swapouts"] + 2)
    assert machine.swapped_out_bytes(vm, later) == 2 * 16384
    assert machine.swapped_out_bytes(None, later) is None
    monkeypatch.setattr(machine, "_run", lambda cmd, timeout=30: None)
    assert machine.vm_counters() is None
