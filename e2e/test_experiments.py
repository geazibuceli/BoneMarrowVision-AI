"""Exercise local experiment discovery and detail rendering against the real API."""

import json

import pytest
from playwright.sync_api import expect


@pytest.mark.parametrize("width", [1440, 360], ids=["desktop", "mobile"])
def test_local_experiment_details(application, browser, width):
    directory = application["run"] / "data/experiments/paired-fixture"
    (directory / "classifier").mkdir(parents=True)
    (directory / "status.json").write_text('{"stage":"training_classifier"}')
    (directory / "config.json").write_text('{"seed":42,"classifier":{"epochs":2}}')
    history = [{"epoch": 1, "train_loss": 0.8, "validation_loss": 0.6, "validation_macro_f1": 0.7},
               {"epoch": 2, "train_loss": 0.5, "validation_loss": 0.4, "validation_macro_f1": 0.8}]
    (directory / "classifier/best.history.json").write_text(json.dumps(history))
    context = browser.new_context(viewport={"width": width, "height": 900})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.goto(f'{application["url"]}/login')
        page.get_by_label("Email", exact=True).fill("admin@example.com")
        page.get_by_label("Password", exact=True).fill(application["password"])
        page.get_by_role("button", name="Sign in", exact=True).click()
        page.get_by_role("link", name="Experiments", exact=True).click()
        page.get_by_role("link", name="paired-fixture", exact=True).click()
        expect(page.get_by_role("heading", name="Experiment details")).to_be_visible()
        expect(page.get_by_text("Stage: training classifier", exact=True)).to_be_visible()
        expect(page.get_by_text("Test comparison is not available yet.")).to_be_visible()
        expect(page.get_by_role("img", name="GAN augmented: validation macro F1", exact=False)).to_be_visible()
        page.get_by_text("View epoch metrics", exact=True).click()
        expect(page.get_by_role("cell", name="80.00%", exact=True)).to_be_visible()

        metrics = {"test_accuracy": 0.86, "test_macro_precision": 0.75,
                   "test_macro_f1": 0.73, "test_loss": 0.58}
        comparison = {"gan_augmented": {**metrics, "test_macro_f1": 0.72},
                      "historical_baseline": metrics, "limitations": "Historical only."}
        (directory / "comparison.json").write_text(json.dumps(comparison))
        (directory / "status.json").write_text('{"stage":"completed"}')
        (directory / "test_report.json").write_text(json.dumps({
            "class_names": ["class-a", "class-b"], "confusion_matrix": [[8, 2], [1, 9]],
        }))
        page.get_by_role("button", name="Refresh experiment").click()
        expect(page.get_by_text("Stage: completed", exact=True)).to_be_visible()
        expect(page.get_by_text("Historical comparison only.", exact=False)).to_be_visible()
        expect(page.get_by_role("cell", name="72.00%", exact=True)).to_be_visible()
        expect(page.get_by_role("row", name="class-a 88.89% 80.00% 84.21% 10")).to_be_visible()

        comparison.update({"comparison_type": "matched_real_only_control",
                           "real_only_control": metrics,
                           "delta_gan_minus_control": {"test_macro_f1": -0.01},
                           "limitations": "Single-seed matched comparison."})
        (directory / "comparison.json").write_text(json.dumps(comparison))
        page.get_by_role("button", name="Refresh experiment").click()
        expect(page.get_by_text("Macro F1 change, GAN minus control: -1.00 percentage points.")).to_be_visible()
        page.get_by_text("Training configuration", exact=True).click()
        expect(page.locator("pre").filter(has_text='"seed": 42')).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.reload()
        expect(page.get_by_role("heading", name="Experiment details")).to_be_visible()
        page.get_by_role("link", name="Back to experiments").click()
        expect(page.get_by_role("link", name="paired-fixture", exact=True)).to_be_visible()
        page.goto(f'{application["url"]}/experiments/runs/missing')
        expect(page.get_by_role("alert")).to_have_text("Experiment run not found")
        assert not errors, errors
    except Exception:
        page.screenshot(path=str(application["run"] / "failure.png"), full_page=True)
        raise
    finally:
        context.close()
