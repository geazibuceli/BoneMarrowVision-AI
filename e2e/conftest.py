"""Real browser/API integration with disposable data and a synthetic checkpoint."""

import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / '.e2e-artifacts'


class Application(dict):
    def __repr__(self):
        return 'Application(isolated test environment)'


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


@pytest.fixture
def application():
    api_port, frontend_port = free_port(), free_port()
    while frontend_port == api_port:
        frontend_port = free_port()
    api_url = f'http://127.0.0.1:{api_port}'
    frontend_url = f'http://127.0.0.1:{frontend_port}'
    run = ARTIFACTS / uuid4().hex
    run.mkdir(parents=True)
    for directory in ('models', 'configs', 'uploads', 'inputs'):
        (run / directory).mkdir()
    import torch
    from bone_marrow_vision.models import build_model
    from PIL import Image

    torch.manual_seed(42)
    model = build_model('baseline_cnn', 3, pretrained=False)
    torch.save({'architecture': 'baseline_cnn', 'class_names': ['test-a', 'test-b', 'test-c'],
                'image_size': 64, 'model_state': model.state_dict()}, run / 'models/fixture.pt')
    (run / 'configs/fixture.yaml').write_text('architecture: baseline_cnn\n', encoding='utf-8')
    Image.new('RGB', (64, 64), (90, 40, 140)).save(run / 'cell.png')
    password = secrets.token_urlsafe(24)
    bootstrap = secrets.token_urlsafe(48)
    env = {**os.environ, 'APP_ENVIRONMENT': 'test', 'PROJECT_ROOT': str(run),
           'DATABASE_URL': f'sqlite:///{(run / "app.db").as_posix()}',
           'JWT_SECRET': secrets.token_urlsafe(48), 'ADMIN_BOOTSTRAP_TOKEN': bootstrap,
           'MODEL_DIR': str(run / 'models'), 'CONFIG_DIR': str(run / 'configs'),
           'UPLOAD_DIR': str(run / 'uploads'), 'INFERENCE_TEMP_DIR': str(run / 'inputs'),
           'MPLCONFIGDIR': str(run / 'matplotlib'), 'AUTH_RATE_LIMIT': '200',
           'PREDICTION_RATE_LIMIT': '200', 'TRUST_PROXY_HEADERS': 'false',
           'BMV_BACKEND_URL': api_url, 'VITE_API_URL': '/api/v1',
           'CORS_ORIGINS': frontend_url}
    processes = []
    handles = []
    try:
        subprocess.run([sys.executable, '-m', 'alembic', 'upgrade', 'head'],
                       cwd=ROOT / 'backend', env=env, check=True, timeout=60)
        for label, command, cwd in (
            ('api', [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(api_port)], ROOT / 'backend'),
            ('frontend', [shutil.which('node'), 'node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', str(frontend_port)], ROOT / 'frontend'),
        ):
            handle = (run / f'{label}.log').open('w', encoding='utf-8')
            handles.append(handle)
            processes.append(subprocess.Popen(command, cwd=cwd, env=env, stdout=handle, stderr=handle,
                                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
        for url in (f'{api_url}/api/v1/health/ready', frontend_url):
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError(f'A test server exited. See {run}')
                try:
                    if httpx.get(url, timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError(f'Test server did not become ready: {url}')
        with httpx.Client(base_url=f'{api_url}/api/v1', timeout=60) as client:
            client.post('/auth/register', json={'email': 'admin@example.com', 'password': password,
                        'full_name': 'Test Admin', 'admin_bootstrap_token': bootstrap}).raise_for_status()
            login = client.post('/auth/login', data={'username': 'admin@example.com', 'password': password})
            login.raise_for_status()
            headers = {'Authorization': f'Bearer {login.json()["access_token"]}'}
            yield Application(run=run, password=password, client=client, admin_headers=headers, url=frontend_url)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        for handle in handles:
            handle.close()


@pytest.fixture
def browser():
    with sync_playwright() as playwright:
        channel = os.environ.get('E2E_BROWSER_CHANNEL')
        browser = playwright.chromium.launch(headless=True, **({'channel': channel} if channel else {}))
        try:
            yield browser
        finally:
            browser.close()
