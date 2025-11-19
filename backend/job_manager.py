from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Literal

JobType = Literal["image", "video"]
JobStatus = Literal["pending", "running", "paused", "completed", "failed", "cancelled"]


class JobCancelledError(Exception):
    """Exception levée lorsqu'un job est annulé."""


class JobPauseRequested(Exception):
    """Exception levée lorsqu'un job doit être mis en pause."""


class JobManager:
    """
    Gestionnaire de jobs persistants avec file FIFO et reprise automatique.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.jobs_dir = self.storage_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.queue: list[str] = []
        self.executors: dict[str, Callable[[Dict[str, Any], "JobManager"], None]] = {}
        self.control_flags: Dict[str, Dict[str, bool]] = {}

        self.lock = threading.Lock()
        self.new_job_event = threading.Event()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self._load_jobs()

    # ------------------------------------------------------------------
    # Chargement / persistance
    # ------------------------------------------------------------------
    def _job_file(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _load_jobs(self) -> None:
        for file in self.jobs_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                job_id = data["id"]
                self.jobs[job_id] = data
            except Exception:
                continue

        # Reconstituer la file : jobs pending + anciens running/paused
        pending_jobs = [
            job
            for job in self.jobs.values()
            if job["status"] in ("pending", "running", "paused")
        ]
        pending_jobs.sort(key=lambda j: j.get("created_at", 0))

        for job in pending_jobs:
            if job["status"] == "running":
                job["status"] = "pending"
            self.queue.append(job["id"])
            self._save_job(job)

    def _save_job(self, job: Dict[str, Any]) -> None:
        job["updated_at"] = time.time()
        file = self._job_file(job["id"])
        file.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Gestion du worker
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_loop, name="JobWorker", daemon=True
        )
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.new_job_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def register_executor(
        self, job_type: JobType, executor: Callable[[Dict[str, Any], "JobManager"], None]
    ) -> None:
        self.executors[job_type] = executor

    def _pop_next_job(self) -> str | None:
        with self.lock:
            while self.queue:
                job_id = self.queue.pop(0)
                job = self.jobs.get(job_id)
                if not job:
                    continue
                if job["status"] in ("pending",):
                    return job_id
        return None

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            job_id = self._pop_next_job()
            if not job_id:
                self.new_job_event.wait(timeout=1)
                self.new_job_event.clear()
                continue

            job = self.jobs.get(job_id)
            if not job:
                continue

            executor = self.executors.get(job["type"])
            if not executor:
                job["status"] = "failed"
                job["error"] = f"Aucun executor pour le type {job['type']}"
                self._save_job(job)
                continue

            job["status"] = "running"
            job["started_at"] = time.time()
            self.control_flags[job_id] = {"pause": False, "cancel": False}
            self._save_job(job)

            try:
                executor(job, self)
                if job["status"] == "running":
                    job["status"] = "completed"
            except JobPauseRequested:
                job["status"] = "paused"
            except JobCancelledError:
                job["status"] = "cancelled"
            except Exception as err:  # pylint: disable=broad-except
                job["status"] = "failed"
                job["error"] = str(err)
            finally:
                job.setdefault("finished_at", time.time())
                self._save_job(job)
                self.control_flags.pop(job_id, None)

    # ------------------------------------------------------------------
    # Création / gestion des jobs
    # ------------------------------------------------------------------
    def create_job(
        self,
        job_type: JobType,
        payload: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
        total_steps: int | None = None,
    ) -> Dict[str, Any]:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        timestamp = time.time()
        total = (
            total_steps
            or payload.get("image_count")
            or payload.get("num_frames")
            or 1
        )
        job = {
            "id": job_id,
            "type": job_type,
            "status": "pending",
            "payload": payload,
            "metadata": metadata or {},
            "progress": {"current": 0, "total": total},
            "result": None,
            "error": None,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self.lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self._save_job(job)
        self.new_job_event.set()
        return job

    def list_jobs(self) -> list[Dict[str, Any]]:
        with self.lock:
            return list(self.jobs.values())

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        return self.jobs.get(job_id)

    def update_progress(
        self, job_id: str, current: int, total: int | None = None, message: str | None = None
    ) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        progress = job.setdefault("progress", {})
        progress["current"] = current
        if total is not None:
            progress["total"] = total
        if message is not None:
            progress["message"] = message
        self._save_job(job)

    def set_result(self, job_id: str, result: Dict[str, Any]) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        job["result"] = result
        self._save_job(job)

    # ------------------------------------------------------------------
    # Contrôle (pause, reprise, annulation)
    # ------------------------------------------------------------------
    def request_pause(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job["status"] not in ("pending", "running"):
            return False

        if job["status"] == "pending":
            with self.lock:
                if job_id in self.queue:
                    self.queue.remove(job_id)
            job["status"] = "paused"
            self._save_job(job)
            return True

        # job running
        flags = self.control_flags.setdefault(job_id, {"pause": False, "cancel": False})
        flags["pause"] = True
        return True

    def resume_job(self, job_id: str, prioritize: bool = False) -> bool:
        job = self.jobs.get(job_id)
        if not job or job["status"] not in ("paused", "failed"):
            return False
        job["status"] = "pending"
        job["error"] = None
        job["progress"]["message"] = "En attente de reprise"
        with self.lock:
            if prioritize:
                self.queue.insert(0, job_id)
            else:
                self.queue.append(job_id)
            self._save_job(job)
        self.new_job_event.set()
        return True

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job["status"] in ("completed", "cancelled"):
            return False

        if job["status"] == "pending":
            with self.lock:
                if job_id in self.queue:
                    self.queue.remove(job_id)
            job["status"] = "cancelled"
            self._save_job(job)
            return True

        flags = self.control_flags.setdefault(job_id, {"pause": False, "cancel": False})
        flags["cancel"] = True
        return True

    def delete_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job["status"] == "running":
            return False

        with self.lock:
            self.jobs.pop(job_id, None)
            if job_id in self.queue:
                self.queue.remove(job_id)
        self.control_flags.pop(job_id, None)
        try:
            self._job_file(job_id).unlink(missing_ok=True)
        except Exception:
            pass
        return True

    def clear_completed(self) -> int:
        removed = 0
        for job_id in list(self.jobs.keys()):
            job = self.jobs[job_id]
            if job["status"] in ("completed", "cancelled", "failed"):
                try:
                    self._job_file(job_id).unlink(missing_ok=True)
                except Exception:
                    pass
                with self.lock:
                    self.jobs.pop(job_id, None)
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Helpers pour les executors
    # ------------------------------------------------------------------
    def raise_if_interrupted(self, job_id: str) -> None:
        flags = self.control_flags.get(job_id)
        if not flags:
            return
        if flags.get("cancel"):
            raise JobCancelledError
        if flags.get("pause"):
            raise JobPauseRequested


