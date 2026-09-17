import asyncio
import os
from pathlib import Path

import anyio

from app.retention import cleanup_expired_artifacts


def test_cleanup_removes_only_expired_generated_artifacts(tmp_path: Path):
    upload_dir = tmp_path / "uploads"
    gradcam_dir = upload_dir / "gradcam"
    temp_dir = tmp_path / "temporary"
    gradcam_dir.mkdir(parents=True)
    temp_dir.mkdir()
    old_time = 1_000_000.0
    current_time = old_time + 31 * 24 * 60 * 60

    old_gradcam = gradcam_dir / "bone-marrow-input-old.png"
    recent_gradcam = gradcam_dir / "bone-marrow-input-recent.png"
    unrelated_gradcam = gradcam_dir / "manual-review.png"
    old_temp = temp_dir / "bone-marrow-input-old.jpg"
    recent_temp = temp_dir / "bone-marrow-input-recent.jpg"
    for path in (old_gradcam, recent_gradcam, unrelated_gradcam, old_temp, recent_temp):
        path.write_bytes(b"artifact")
    os.utime(old_gradcam, (old_time, old_time))
    os.utime(old_temp, (old_time, old_time))
    os.utime(recent_gradcam, (current_time, current_time))
    os.utime(recent_temp, (current_time, current_time))

    removed = cleanup_expired_artifacts(
        upload_dir=upload_dir,
        inference_temp_dir=temp_dir,
        gradcam_retention_days=30,
        temp_retention_hours=24,
        now=current_time,
    )

    assert removed == {"gradcam": 1, "temporary_inputs": 1}
    assert not old_gradcam.exists()
    assert recent_gradcam.exists()
    assert unrelated_gradcam.exists()
    assert not old_temp.exists()
    assert recent_temp.exists()


def test_lifespan_removes_artifacts_created_after_startup_and_stops(tmp_path, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(main.settings, "inference_temp_dir", tmp_path / "inputs")
    monkeypatch.setattr(main.settings, "retention_cleanup_interval_seconds", 0.01)
    directory = tmp_path / "gradcam"
    directory.mkdir()
    artifact = directory / "bone-marrow-input-expired.png"

    async def scenario():
        async with main.lifespan(main.app):
            artifact.write_bytes(b"expired")
            os.utime(artifact, (1, 1))
            with anyio.fail_after(5):
                while artifact.exists():
                    await anyio.sleep(0.01)
        artifact.write_bytes(b"keep after shutdown")
        os.utime(artifact, (1, 1))
        await anyio.sleep(0.05)
        assert artifact.exists()

    asyncio.run(scenario())


def test_scheduled_cleanup_recovers_from_failure(caplog):
    from app.retention import run_periodic_cleanup

    calls = []

    def cleanup():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary filesystem failure")
        return {"gradcam": 0, "temporary_inputs": 0}

    async def scenario():
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(run_periodic_cleanup, cleanup, 0.01)
                while len(calls) < 2:
                    await anyio.sleep(0.01)
                tasks.cancel_scope.cancel()

    asyncio.run(scenario())
    assert "Scheduled artifact cleanup failed" in caplog.text
