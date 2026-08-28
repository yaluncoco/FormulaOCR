"""Small cross-process file lock used by model installers.

Only exclusive locks are needed here. Keeping this implementation local avoids
pulling the broad ``filelock`` package (including its optional SQLite-backed
read/write lock) into the frozen application.
"""

from __future__ import annotations

import errno
import os
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Callable


class InterProcessFileLock:
    def __init__(
        self,
        lock_file: str | os.PathLike[str],
        *,
        timeout: float = -1,
        poll_interval: float = 0.1,
        on_wait: Callable[[], None] | None = None,
        on_wait_interval: float = 0.5,
    ) -> None:
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.poll_interval = max(0.01, poll_interval)
        self.on_wait = on_wait
        self.on_wait_interval = max(self.poll_interval, on_wait_interval)
        self._fd: int | None = None

    def __enter__(self) -> InterProcessFileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_file.is_symlink():
            raise OSError(f"拒绝使用链接锁文件：{self.lock_file}")

        started = time.monotonic()
        last_wait_notification = started - self.on_wait_interval
        while True:
            fd = os.open(self.lock_file, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _try_lock(fd)
            except OSError as exc:
                os.close(fd)
                if not _is_lock_contention(exc):
                    raise
                now = time.monotonic()
                if self.timeout >= 0 and now - started >= self.timeout:
                    raise TimeoutError(f"等待模型下载锁超时：{self.lock_file}") from exc
                if (
                    self.on_wait is not None
                    and now - last_wait_notification >= self.on_wait_interval
                ):
                    # The callback is allowed to raise. Downloaders use this
                    # path to make a GUI cancellation request interrupt a wait
                    # for another FormulaOCR process instead of appearing hung.
                    self.on_wait()
                    last_wait_notification = now
                time.sleep(self.poll_interval)
                continue
            self._fd = fd
            return

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            _unlock(fd)
        finally:
            os.close(fd)


def _try_lock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _is_lock_contention(exc: OSError) -> bool:
    return exc.errno in {
        errno.EACCES,
        errno.EAGAIN,
        getattr(errno, "EDEADLK", errno.EACCES),
    }
