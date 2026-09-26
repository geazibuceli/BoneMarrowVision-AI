import sqlite3

import pytest
from playwright.sync_api import expect


@pytest.mark.parametrize('width', [1440, 360], ids=['desktop', 'mobile'])
def test_analysis_history_and_session(application, browser, width):
    context = browser.new_context(viewport={'width': width, 'height': 900}, locale='pt-BR')
    page = context.new_page()
    page.set_default_timeout(15000)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    run = application['run']
    try:
        page.goto(f'{application["url"]}/register')
        page.get_by_label('Email', exact=True).fill('user@example.com')
        page.get_by_label('Full name', exact=True).fill('Browser User')
        page.get_by_label('Password', exact=True).fill(application['password'])
        page.get_by_role('button', name='Register', exact=True).click()
        expect(page.get_by_role('heading', name='Dashboard', exact=True)).to_be_visible()
        page.get_by_role('link', name='History', exact=True).click()
        expect(page.get_by_text('No predictions recorded for this account.')).to_be_visible()
        page.get_by_role('link', name='Analyze', exact=True).click()
        page.locator('input[type=file]').set_input_files(str(run / 'cell.png'))
        page.get_by_role('button', name='Run analysis').click()
        expect(page.get_by_role('alert')).to_have_text('No active model is available for inference')

        client = application['client']
        headers = application['admin_headers']
        registered = client.post('/models/register', headers=headers, json={
            'name': 'Synthetic test model', 'version': 'test', 'architecture': 'baseline_cnn',
            'checkpoint_path': 'models/fixture.pt', 'config_path': 'configs/fixture.yaml'})
        registered.raise_for_status()
        client.post(f'/models/{registered.json()["id"]}/activate', headers=headers).raise_for_status()
        page.get_by_role('button', name='Run analysis').click()
        expect(page.get_by_role('link', name='Open saved result')).to_be_visible(timeout=60000)
        prediction = page.locator('#result-heading').inner_text()
        assert prediction in ('test-a', 'test-b', 'test-c')
        image = page.get_by_alt_text('Grad-CAM activation overlay for the prediction')
        expect(image).to_be_visible()
        expect(image).to_have_js_property('complete', True)
        assert image.evaluate('(image) => image.naturalWidth > 0')
        assert not list((run / 'inputs').iterdir())
        page.get_by_role('link', name='Open saved result').click()
        detail_url = page.url
        page.reload()
        expect(page.locator('#result-heading')).to_have_text(prediction)
        # English month abbreviations remain English under a Portuguese browser locale.
        assert any(month in page.locator('time').inner_text() for month in
                   ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        with sqlite3.connect(run / 'app.db') as database:
            for _ in range(30):
                database.execute('INSERT INTO predictions (user_id,model_id,predicted_class,confidence,top_predictions_json,gradcam_image_path,low_confidence,created_at) SELECT user_id,model_id,predicted_class,confidence,top_predictions_json,NULL,low_confidence,created_at FROM predictions WHERE id=1')
        page.get_by_role('link', name='Back to history').click()
        expect(page.locator('tbody tr')).to_have_count(25)
        page.get_by_role('button', name='Next', exact=True).click()
        expect(page.locator('tbody tr')).to_have_count(6)
        page.get_by_role('link', name='View prediction 1', exact=True).click()
        expect(page.locator('#result-heading')).to_have_text(prediction)
        page.get_by_role('link', name='Back to history').click()
        expect(page).to_have_url(f'{application["url"]}/history?page=2')
        page.reload()
        expect(page.locator('tbody tr')).to_have_count(6)
        page.get_by_label('Class (exact name)').fill('missing-class')
        page.get_by_role('button', name='Apply filters').click()
        expect(page.locator('tbody tr')).to_have_count(0)
        expect(page.get_by_text('Page 1', exact=True)).to_be_visible()
        page.get_by_label('Class (exact name)').fill(prediction)
        page.get_by_role('button', name='Apply filters').click()
        expect(page.locator('tbody tr')).to_have_count(25)
        with page.expect_download() as download_info:
            page.get_by_role('button', name='Export filtered CSV').click()
        download = download_info.value
        assert download.suggested_filename == 'predictions.csv'
        import csv
        from pathlib import Path
        with Path(download.path()).open(encoding='utf-8', newline='') as stream:
            exported = list(csv.DictReader(stream))
        assert len(exported) == 31
        assert all(row['class'] == prediction for row in exported)
        page.get_by_role('button', name='Next', exact=True).click()
        page.get_by_role('link', name='View prediction 1', exact=True).click()
        page.get_by_role('link', name='Back to history').click()
        expect(page.get_by_label('Class (exact name)')).to_have_value(prediction)
        expect(page.locator('tbody tr')).to_have_count(6)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.get_by_role('link', name='Analyze', exact=True).click()
        page.locator('input[type=file]').set_input_files({'name': 'bad.txt', 'mimeType': 'text/plain', 'buffer': b'bad'})
        expect(page.get_by_role('alert')).to_have_text('Choose a valid PNG or JPEG image.')
        expect(page.get_by_role('button', name='Run analysis')).to_be_disabled()
        page.locator('input[type=file]').set_input_files({'name': 'bad.png', 'mimeType': 'image/png', 'buffer': b'bad'})
        page.get_by_role('button', name='Run analysis').click()
        expect(page.get_by_role('alert')).to_have_text('The uploaded file is not a valid PNG or JPEG image')
        page.evaluate("localStorage.setItem('bone-marrow-vision:token', 'expired-token')")
        page.reload()
        expect(page.get_by_role('heading', name='Welcome back')).to_be_visible()
        # The admin is a different owner, so even this account cannot read the result.
        page.get_by_label('Email', exact=True).fill('admin@example.com')
        page.get_by_label('Password', exact=True).fill(application['password'])
        page.get_by_role('button', name='Sign in', exact=True).click()
        expect(page.get_by_role('heading', name='Inspect an image')).to_be_visible()
        page.goto(detail_url)
        expect(page.get_by_role('alert')).to_contain_text('Prediction not found')
        assert not errors, errors
    except Exception:
        page.screenshot(path=str(run / 'failure.png'), full_page=True)
        raise
    finally:
        context.close()


@pytest.mark.parametrize('width', [1440, 360], ids=['desktop', 'mobile'])
def test_password_change_and_recovery_feedback(application, browser, width):
    context = browser.new_context(viewport={'width': width, 'height': 900})
    page = context.new_page()
    try:
        page.goto(f'{application["url"]}/login')
        page.get_by_role('link', name='Forgot your password?').click()
        expect(page.get_by_role('heading', name='Recover your account')).to_be_visible()
        page.get_by_label('Email').fill('admin@example.com')
        page.get_by_role('button', name='Send reset link').click()
        expect(page.get_by_role('alert')).to_contain_text('Email recovery is not configured')
        page.goto(f'{application["url"]}/login')
        page.get_by_label('Email', exact=True).fill('admin@example.com')
        page.get_by_label('Password', exact=True).fill(application['password'])
        page.get_by_role('button', name='Sign in', exact=True).click()
        page.get_by_role('link', name='Password', exact=True).click()
        expect(page.get_by_role('heading', name='Change password', exact=True)).to_be_visible()
        page.get_by_label('Current password', exact=True).fill(application['password'])
        page.get_by_label('New password', exact=True).fill('Changed-password-123!')
        page.get_by_label('Confirm new password', exact=True).fill('Changed-password-123!')
        page.get_by_role('button', name='Update password').click()
        expect(page.get_by_text('Password updated. Sign in again on all devices.')).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.goto(f'{application["url"]}/login')
        page.get_by_label('Email', exact=True).fill('admin@example.com')
        page.get_by_label('Password', exact=True).fill('Changed-password-123!')
        page.get_by_role('button', name='Sign in', exact=True).click()
        expect(page.get_by_role('heading', name='Dashboard', exact=True)).to_be_visible()
        response = application['client'].get('/auth/me', headers=application['admin_headers'])
        assert response.status_code == 401
    except Exception:
        page.screenshot(path=str(application['run'] / 'failure.png'), full_page=True)
        raise
    finally:
        context.close()
