"""Durable task records. Each thread owns its SQLite connection."""

import json
import os
import sqlite3
import stat
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    id: int
    source: Path
    target: Path
    kind: str
    params: dict
    source_size: int
    source_mtime: int


class ConflictError(Exception):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                params TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                source_mtime INTEGER NOT NULL,
                batch TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated REAL NOT NULL,
                output_size INTEGER,
                output_mtime INTEGER,
                publish_dev INTEGER,
                publish_ino INTEGER
            );
            CREATE INDEX IF NOT EXISTS task_dispatch ON tasks(batch,status,id);
            CREATE TABLE IF NOT EXISTS speech_checkpoints (
                task_id INTEGER NOT NULL, name TEXT NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(task_id,name)
            );
            CREATE TABLE IF NOT EXISTS speech_artifacts (
                task_id INTEGER NOT NULL, path TEXT NOT NULL,
                complete INTEGER NOT NULL DEFAULT 0,
                size INTEGER, mtime INTEGER, dev INTEGER, ino INTEGER,
                PRIMARY KEY(task_id,path)
            );
        """)

    def close(self):
        self.db.close()

    def commit(self):
        self.db.commit()

    def enqueue(
        self, source: Path, target: Path, kind: str, params: dict, batch: str, retry_failed: bool = False
    ) -> str:
        if kind == "transcribe":
            return self._enqueue_speech(source, target, params, batch, retry_failed)
        snapshot = source.stat()
        encoded = json.dumps(params, sort_keys=True, ensure_ascii=False)
        row = self.db.execute("SELECT * FROM tasks WHERE target=?", (str(target),)).fetchone()
        try:
            artifact = target.lstat()
        except FileNotFoundError:
            artifact = None
        same = row is not None and (
            row["source"] == str(source)
            and row["kind"] == kind
            and row["params"] == encoded
            and row["source_size"] == snapshot.st_size
            and row["source_mtime"] == snapshot.st_mtime_ns
        )
        if artifact is not None:
            if (
                same
                and row["status"] == "succeeded"
                and stat.S_ISREG(artifact.st_mode)
                and row["output_size"] == artifact.st_size
                and row["output_mtime"] == artifact.st_mtime_ns
            ):
                self.db.execute("UPDATE tasks SET batch=? WHERE id=?", (batch, row["id"]))
                return "skipped"
            raise ConflictError(f"输出冲突，文件未改动；请使用其他输出目录：{target}")
        if row is not None and row["source"] != str(source):
            raise ConflictError(f"输出路径已属于另一输入：{target}")
        status = "failed" if same and row["status"] == "failed" and not retry_failed else "pending"
        if row is None:
            self.db.execute(
                """
                INSERT INTO tasks(source,target,kind,params,source_size,source_mtime,batch,status,updated)
                VALUES(?,?,?,?,?,?,?,?,?)
            """,
                (
                    str(source),
                    str(target),
                    kind,
                    encoded,
                    snapshot.st_size,
                    snapshot.st_mtime_ns,
                    batch,
                    status,
                    time.time(),
                ),
            )
        else:
            self.db.execute(
                """
                UPDATE tasks SET params=?,source_size=?,source_mtime=?,batch=?,status=?,
                    error=CASE WHEN ?='failed' THEN error ELSE NULL END,
                    updated=?,output_size=NULL,output_mtime=NULL,publish_dev=NULL,publish_ino=NULL
                WHERE id=?
            """,
                (
                    encoded,
                    snapshot.st_size,
                    snapshot.st_mtime_ns,
                    batch,
                    status,
                    status,
                    time.time(),
                    row["id"],
                ),
            )
        return status

    def claim(self, batch: str, limit: int, *, speech: bool | None = None) -> list[Task]:
        if limit <= 0:
            return []
        with self.db:
            kind_filter = (
                "" if speech is None else (" AND kind='transcribe'" if speech else " AND kind!='transcribe'")
            )
            rows = self.db.execute(
                "SELECT * FROM tasks WHERE batch=? AND status='pending'"
                + kind_filter
                + " ORDER BY id LIMIT ?",
                (batch, limit),
            ).fetchall()
            self.db.executemany(
                "UPDATE tasks SET status='running',attempts=attempts+1,updated=? WHERE id=?",
                [(time.time(), row["id"]) for row in rows],
            )
        return [
            Task(
                row["id"],
                Path(row["source"]),
                Path(row["target"]),
                row["kind"],
                json.loads(row["params"]),
                row["source_size"],
                row["source_mtime"],
            )
            for row in rows
        ]

    def counts(self, batch: str | None = None) -> dict[str, int]:
        where, args = (" WHERE batch=?", (batch,)) if batch else ("", ())
        result = dict.fromkeys(("pending", "running", "succeeded", "failed"), 0)
        result.update(
            {
                row[0]: row[1]
                for row in self.db.execute(
                    "SELECT status,COUNT(*) FROM tasks" + where + " GROUP BY status", args
                )
            }
        )
        return result

    def failures(self, limit: int = 20, batch: str | None = None):
        where = "status='failed'" + (" AND batch=?" if batch else "")
        args = (batch, limit) if batch else (limit,)
        return self.db.execute(
            f"SELECT source,kind,error FROM tasks WHERE {where} ORDER BY id LIMIT ?", args
        ).fetchall()

    def publishing(self, task: Task, artifact: os.stat_result):
        with self.db:
            self.db.execute(
                "UPDATE tasks SET publish_dev=?,publish_ino=?,updated=? WHERE id=?",
                (artifact.st_dev, artifact.st_ino, time.time(), task.id),
            )

    def succeed(self, task: Task):
        artifact = task.target.stat()
        with self.db:
            self.db.execute(
                """
                UPDATE tasks SET status='succeeded',error=NULL,output_size=?,output_mtime=?,
                    publish_dev=NULL,publish_ino=NULL,updated=? WHERE id=?
            """,
                (artifact.st_size, artifact.st_mtime_ns, time.time(), task.id),
            )

    def finish_error(self, task: Task, error: str, cancelled: bool = False):
        self._clean_publication(task.id)
        with self.db:
            self.db.execute(
                """
                UPDATE tasks SET status=?,error=?,publish_dev=NULL,publish_ino=NULL,updated=? WHERE id=?
            """,
                ("pending" if cancelled else "failed", error[-8000:], time.time(), task.id),
            )

    def _clean_publication(self, task_id: int):
        for item in self.db.execute(
            "SELECT path,dev,ino FROM speech_artifacts WHERE task_id=? AND complete=0", (task_id,)
        ).fetchall():
            target = Path(item["path"])
            try:
                current = target.lstat()
                if (current.st_dev, current.st_ino) == (item["dev"], item["ino"]):
                    target.unlink()
            except FileNotFoundError:
                pass
        self.db.execute("DELETE FROM speech_artifacts WHERE task_id=? AND complete=0", (task_id,))
        row = self.db.execute(
            "SELECT target,publish_dev,publish_ino FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        if row["publish_ino"] is None:
            return
        target = Path(row["target"])
        try:
            current = target.lstat()
            if (current.st_dev, current.st_ino) == (row["publish_dev"], row["publish_ino"]):
                target.unlink()
        except FileNotFoundError:
            pass

    def recover(self):
        # Called only while the exclusive runner lock is held.
        for row in self.db.execute("SELECT id FROM tasks WHERE status='running'").fetchall():
            self._clean_publication(row["id"])
        with self.db:
            self.db.execute(
                """
                UPDATE tasks SET status='pending',error=NULL,publish_dev=NULL,publish_ino=NULL,
                    updated=? WHERE status='running'
            """,
                (time.time(),),
            )

    @staticmethod
    def speech_targets(target: Path) -> tuple[Path, ...]:
        return target, target.with_suffix(".txt"), target.with_suffix(".srt")

    def artifact_ready(self, task_id: int, target: Path) -> bool:
        try:
            current = target.lstat()
        except FileNotFoundError:
            return False
        row = self.db.execute(
            "SELECT * FROM speech_artifacts WHERE task_id=? AND path=?", (task_id, str(target))
        ).fetchone()
        if not (
            row
            and row["complete"]
            and stat.S_ISREG(current.st_mode)
            and (row["size"], row["mtime"]) == (current.st_size, current.st_mtime_ns)
        ):
            raise ConflictError(f"输出冲突，文件未改动：{target}")
        return True

    def _enqueue_speech(self, source, target, params, batch, retry_failed):
        snapshot = source.stat()
        encoded = json.dumps(params, sort_keys=True, ensure_ascii=False)
        row = self.db.execute("SELECT * FROM tasks WHERE target=?", (str(target),)).fetchone()
        same = row is not None and (
            row["source"] == str(source)
            and row["kind"] == "transcribe"
            and row["params"] == encoded
            and (row["source_size"], row["source_mtime"]) == (snapshot.st_size, snapshot.st_mtime_ns)
        )
        if row is not None and row["source"] != str(source):
            raise ConflictError(f"输出路径已属于另一输入：{target}")
        ready = []
        for path in self.speech_targets(target):
            if not same:
                if path.exists() or path.is_symlink():
                    raise ConflictError(f"输出冲突，请使用其他输出目录：{path}")
                ready.append(False)
            else:
                ready.append(self.artifact_ready(row["id"], path))
        status = (
            "succeeded"
            if all(ready)
            else ("failed" if same and row["status"] == "failed" and not retry_failed else "pending")
        )
        if row is None:
            self.db.execute(
                """INSERT INTO tasks(source,target,kind,params,source_size,source_mtime,batch,status,updated)
                   VALUES(?,?,'transcribe',?,?,?,?,?,?)""",
                (
                    str(source),
                    str(target),
                    encoded,
                    snapshot.st_size,
                    snapshot.st_mtime_ns,
                    batch,
                    status,
                    time.time(),
                ),
            )
        else:
            if not same:
                self.db.execute("DELETE FROM speech_checkpoints WHERE task_id=?", (row["id"],))
                self.db.execute("DELETE FROM speech_artifacts WHERE task_id=?", (row["id"],))
            self.db.execute(
                """UPDATE tasks SET kind='transcribe',params=?,source_size=?,source_mtime=?,batch=?,
                   status=?,error=CASE WHEN ?='failed' THEN error ELSE NULL END,updated=? WHERE id=?""",
                (
                    encoded,
                    snapshot.st_size,
                    snapshot.st_mtime_ns,
                    batch,
                    status,
                    status,
                    time.time(),
                    row["id"],
                ),
            )
        return "skipped" if status == "succeeded" else status

    def checkpoint(self, task_id: int, name: str):
        row = self.db.execute(
            "SELECT data FROM speech_checkpoints WHERE task_id=? AND name=?", (task_id, name)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_checkpoint(self, task_id: int, name: str, data):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO speech_checkpoints VALUES(?,?,?)",
                (task_id, name, json.dumps(data, ensure_ascii=False, allow_nan=False)),
            )

    def artifact_publishing(self, task_id: int, target: Path, identity: os.stat_result):
        with self.db:
            self.db.execute(
                """INSERT OR REPLACE INTO speech_artifacts(task_id,path,complete,dev,ino)
                   VALUES(?,?,0,?,?)""",
                (task_id, str(target), identity.st_dev, identity.st_ino),
            )

    def artifact_complete(self, task_id: int, target: Path):
        current = target.stat()
        with self.db:
            self.db.execute(
                "UPDATE speech_artifacts SET complete=1,size=?,mtime=? WHERE task_id=? AND path=?",
                (current.st_size, current.st_mtime_ns, task_id, str(target)),
            )

    def speech_succeed(self, task: Task):
        if not all(self.artifact_ready(task.id, path) for path in self.speech_targets(task.target)):
            raise ConflictError("转写导出未全部完成")
        self.succeed(task)
