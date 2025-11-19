import time

import pytest

from job_manager import JobManager


def wait_for(condition, timeout=5.0, interval=0.05):
    """Attendre qu'une condition soit vraie ou lever AssertionError."""
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return
        time.sleep(interval)
    raise AssertionError("Condition non atteinte avant expiration du délai.")


def _make_executor(recorded_labels):
    def executor(job_data, manager):
        total = job_data["progress"]["total"]
        label = job_data["metadata"].get("label", job_data["id"])
        for step in range(total):
            manager.raise_if_interrupted(job_data["id"])
            time.sleep(0.05)
            manager.update_progress(
                job_data["id"],
                current=step + 1,
                total=total,
                message=f"{label} [{step + 1}/{total}]",
            )
        manager.set_result(job_data["id"], {"label": label})
        recorded_labels.append(label)

    return executor


@pytest.fixture()
def job_manager(tmp_path):
    storage_dir = tmp_path / "jobs"
    manager = JobManager(storage_dir)
    yield manager
    manager.stop()


def test_jobs_are_executed_in_fifo_order(job_manager):
    executed = []
    job_manager.register_executor("fake", _make_executor(executed))
    job_manager.start()

    first = job_manager.create_job(
        "fake",
        {"image_count": 2},
        metadata={"label": "first"},
        total_steps=2,
    )
    second = job_manager.create_job(
        "fake",
        {"image_count": 3},
        metadata={"label": "second"},
        total_steps=3,
    )

    wait_for(lambda: job_manager.get_job(second["id"])["status"] == "completed", timeout=10)

    assert executed == ["first", "second"]
    assert job_manager.get_job(first["id"])["progress"]["current"] == 2
    assert job_manager.get_job(second["id"])["progress"]["current"] == 3


def test_pause_and_resume_job(job_manager):
    executed = []
    job_manager.register_executor("fake", _make_executor(executed))
    job_manager.start()

    job = job_manager.create_job(
        "fake",
        {"image_count": 4},
        metadata={"label": "pause-me"},
        total_steps=4,
    )

    wait_for(lambda: job_manager.get_job(job["id"])["status"] == "running")

    assert job_manager.request_pause(job["id"]) is True
    wait_for(lambda: job_manager.get_job(job["id"])["status"] == "paused")

    assert job_manager.resume_job(job["id"], prioritize=True) is True
    wait_for(lambda: job_manager.get_job(job["id"])["status"] == "completed")


def test_cancel_pending_job(job_manager):
    executed = []
    job_manager.register_executor("fake", _make_executor(executed))
    job_manager.start()

    job = job_manager.create_job(
        "fake",
        {"image_count": 1},
        metadata={"label": "will-cancel"},
        total_steps=1,
    )
    assert job_manager.cancel_job(job["id"]) is True
    wait_for(lambda: job_manager.get_job(job["id"])["status"] == "cancelled")
    assert executed == []


def test_delete_job(job_manager):
    job = job_manager.create_job(
        "fake",
        {"image_count": 1},
        metadata={"label": "to-delete"},
        total_steps=1,
    )
    # Pending job can be supprimé
    assert job_manager.delete_job(job["id"]) is True
    assert job_manager.get_job(job["id"]) is None

    # Terminated job aussi
    executed = []
    job_manager.register_executor("fake", _make_executor(executed))
    job_manager.start()
    finished = job_manager.create_job(
        "fake",
        {"image_count": 1},
        metadata={"label": "done"},
        total_steps=1,
    )
    wait_for(lambda: job_manager.get_job(finished["id"])["status"] == "completed")
    assert job_manager.delete_job(finished["id"]) is True
    assert job_manager.get_job(finished["id"]) is None

