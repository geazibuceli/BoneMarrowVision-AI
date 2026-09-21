"""Verify administrator model changes and persistent prediction attribution."""

import shutil

import pytest
from playwright.sync_api import expect


@pytest.mark.parametrize("width", [1440, 360], ids=["desktop", "mobile"])
def test_model_registration_activation_and_restore(application, browser, width):
    run = application["run"]
    shutil.copyfile(run / "models/fixture.pt", run / "models/second.pt")
    context = browser.new_context(viewport={"width": width, "height": 900})
    page = context.new_page()
    page.set_default_timeout(15000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    registry = page.get_by_role("region", name="Registered models", exact=True)

    def row(version):
        return registry.get_by_role("row").filter(has=page.get_by_role("cell", name=version, exact=True))

    def fill_registration(version, checkpoint):
        page.get_by_label("Name", exact=True).fill("UI classifier")
        page.get_by_label("Version", exact=True).fill(version)
        page.get_by_label("Architecture", exact=True).select_option("baseline_cnn")
        page.get_by_label("Checkpoint path", exact=True).fill(checkpoint)
        page.get_by_label("Configuration path", exact=True).fill("configs/fixture.yaml")
        page.get_by_label("Metrics (JSON, optional)", exact=True).fill('{"macro_f1":0.73}')

    try:
        page.goto(f'{application["url"]}/login')
        page.get_by_label("Email", exact=True).fill("admin@example.com")
        page.get_by_label("Password", exact=True).fill(application["password"])
        page.get_by_role("button", name="Sign in", exact=True).click()
        page.get_by_role("link", name="Models", exact=True).click()
        fill_registration("1", "models/fixture.pt")
        page.get_by_label("Metrics (JSON, optional)", exact=True).fill("invalid")
        page.get_by_role("button", name="Register model", exact=True).click()
        expect(page.get_by_role("alert")).to_have_text("Metrics must be a valid JSON object.")
        page.get_by_label("Metrics (JSON, optional)", exact=True).fill('{"macro_f1":0.73}')
        page.get_by_role("button", name="Register model", exact=True).click()
        expect(row("1").get_by_role("cell", name="Available", exact=True)).to_be_visible()
        row("1").get_by_role("button", name="Activate", exact=True).click()
        expect(row("1").get_by_role("cell", name="Active", exact=True)).to_be_visible()

        page.get_by_role("link", name="Analyze", exact=True).click()
        page.locator("input[type=file]").set_input_files(str(run / "cell.png"))
        page.get_by_role("button", name="Run analysis").click()
        expect(page.get_by_text("Model: UI classifier · 1", exact=True)).to_be_visible(timeout=60000)
        page.get_by_role("link", name="Open saved result").click()
        prediction_url = page.url

        page.get_by_role("link", name="Models", exact=True).click()
        fill_registration("2", "models/second.pt")
        page.get_by_role("button", name="Register model", exact=True).click()
        row("2").get_by_role("button", name="Activate", exact=True).click()
        expect(row("2").get_by_role("cell", name="Active", exact=True)).to_be_visible()
        expect(row("1").get_by_role("cell", name="Available", exact=True)).to_be_visible()
        page.goto(prediction_url)
        expect(page.get_by_text("Model: UI classifier · 1", exact=True)).to_be_visible()
        page.get_by_role("link", name="Back to history").click()
        expect(page.get_by_role("cell", name="UI classifier · 1", exact=True)).to_be_visible()

        page.get_by_role("link", name="Models", exact=True).click()
        history = page.get_by_role("region", name="Recorded model activations", exact=True)
        history.get_by_role("row").filter(has=page.get_by_role("cell", name="1", exact=True)).get_by_role(
            "button", name="Activate this version",
        ).click()
        expect(row("1").get_by_role("cell", name="Active", exact=True)).to_be_visible()
        (run / "models/second.pt").write_bytes(b"invalid checkpoint")
        row("2").get_by_role("button", name="Activate", exact=True).click()
        expect(page.get_by_role("alert")).to_have_text(
            "Checkpoint is unavailable or incompatible; active model unchanged",
        )
        expect(row("1").get_by_role("cell", name="Active", exact=True)).to_be_visible()
        page.get_by_text("Inspect UI classifier 1", exact=True).click()
        expect(row("1").get_by_text("Checkpoint SHA-256:", exact=False)).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

        page.get_by_role("button", name="Sign out", exact=True).click()
        page.goto(f'{application["url"]}/register')
        page.get_by_label("Email", exact=True).fill("reader@example.com")
        page.get_by_label("Full name", exact=True).fill("Model Reader")
        page.get_by_label("Password", exact=True).fill(application["password"])
        page.get_by_role("button", name="Register", exact=True).click()
        page.get_by_role("link", name="Models", exact=True).click()
        expect(row("1")).to_be_visible()
        expect(page.get_by_role("button", name="Register model", exact=True)).to_have_count(0)
        expect(page.get_by_role("button", name="Activate", exact=True)).to_have_count(0)
        expect(page.get_by_role("heading", name="Activation history", exact=True)).to_have_count(0)
        assert not errors, errors
    except Exception:
        page.screenshot(path=str(run / "failure.png"), full_page=True)
        raise
    finally:
        context.close()
