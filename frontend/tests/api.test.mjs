import assert from 'node:assert/strict';
import test from 'node:test';
import {
  AUTH_UNAUTHORIZED_EVENT,
  ApiError,
  getProtectedMedia,
  normalizeApiUrl,
  readJsonResponse,
  resolveMediaUrl,
} from '../src/api.ts';

test('normalizes API base URLs without leaving a trailing slash', () => {
  assert.equal(normalizeApiUrl(' /api/v1/ '), '/api/v1');
  assert.equal(normalizeApiUrl('https://api.example.test/api/v1///'), 'https://api.example.test/api/v1');
  assert.equal(normalizeApiUrl('   '), '/api/v1');
  assert.equal(normalizeApiUrl('//malicious.example/api/v1'), '/api/v1');
  assert.equal(normalizeApiUrl('javascript:alert(1)'), '/api/v1');
  assert.equal(normalizeApiUrl('/api/v1?token=unsafe'), '/api/v1');
  assert.equal(normalizeApiUrl('https://user:secret@api.example.test/api/v1'), '/api/v1');
});

test('reads successful JSON and reports FastAPI validation details', async () => {
  const success = new Response(JSON.stringify({ ok: true }), {
    headers: { 'content-type': 'application/json' },
  });
  assert.deepEqual(await readJsonResponse(success, 'Request failed'), { ok: true });

  const invalid = new Response(JSON.stringify({ detail: [{ msg: 'Email is invalid' }, { msg: 'Password is short' }] }), {
    status: 422,
    headers: { 'content-type': 'application/json' },
  });
  await assert.rejects(
    () => readJsonResponse(invalid, 'Registration failed'),
    (error) => error instanceof ApiError && error.status === 422 && error.message === 'Email is invalid. Password is short',
  );
});

test('does not expose HTML errors and identifies empty successful responses', async () => {
  const htmlError = new Response('<html><body>private proxy detail</body></html>', {
    status: 502,
    statusText: 'Bad Gateway',
    headers: { 'content-type': 'text/html' },
  });
  await assert.rejects(
    () => readJsonResponse(htmlError, 'Could not load data'),
    (error) => error instanceof ApiError && error.message === 'Could not load data (Bad Gateway)',
  );

  const emptySuccess = new Response('', { status: 200 });
  await assert.rejects(
    () => readJsonResponse(emptySuccess, 'Could not load data'),
    (error) => error instanceof ApiError && error.message.includes('empty or unexpected'),
  );
});

test('notifies the authentication provider when an authenticated request expires', async () => {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
  const fakeWindow = new EventTarget();
  let notifications = 0;
  fakeWindow.addEventListener(AUTH_UNAUTHORIZED_EVENT, () => {
    notifications += 1;
  });
  Object.defineProperty(globalThis, 'window', { configurable: true, value: fakeWindow });

  try {
    const expired = new Response(JSON.stringify({ detail: 'Session expired' }), {
      status: 401,
      headers: { 'content-type': 'application/json' },
    });
    await assert.rejects(
      () => readJsonResponse(expired, 'Could not validate session', true),
      (error) => error instanceof ApiError && error.status === 401,
    );
    assert.equal(notifications, 1);
  } finally {
    if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow);
    else delete globalThis.window;
  }
});

test('resolves only ownership-checked API media paths', () => {
  assert.equal(
    resolveMediaUrl('/api/v1/predictions/7/gradcam', 'https://api.example.test/api/v1', 'https://app.example.test'),
    'https://api.example.test/api/v1/predictions/7/gradcam',
  );
  assert.throws(
    () => resolveMediaUrl('/media/gradcam/image.png', '/api/v1', 'http://localhost:5173'),
    ApiError,
  );
  assert.throws(
    () => resolveMediaUrl('https://malicious.example/media/image.png', '/api/v1', 'https://app.example.test'),
    ApiError,
  );
  assert.throws(
    () => resolveMediaUrl('/api/v10/predictions/7/gradcam', '/api/v1', 'https://app.example.test'),
    ApiError,
  );
});

test('loads protected media with a bearer token', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/v1/predictions/3/gradcam');
    assert.equal(new Headers(options.headers).get('authorization'), 'Bearer test-token');
    return new Response(new Uint8Array([137, 80, 78, 71]), {
      headers: { 'content-type': 'image/png' },
    });
  };

  try {
    const blob = await getProtectedMedia('/api/v1/predictions/3/gradcam', 'test-token');
    assert.equal(blob.type, 'image/png');
    assert.equal(blob.size, 4);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('rejects non-raster responses from the Grad-CAM endpoint', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response('<svg/>', {
    headers: { 'content-type': 'image/svg+xml' },
  });

  try {
    await assert.rejects(
      () => getProtectedMedia('/api/v1/predictions/3/gradcam', 'test-token'),
      (error) => error instanceof ApiError && error.message.includes('invalid Grad-CAM image'),
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
