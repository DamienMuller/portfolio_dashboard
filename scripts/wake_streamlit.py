"""Wake the Streamlit Community Cloud app with a real browser session."""

import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = "https://ai-portfolio-opt-dashboard.streamlit.app/"
WAKE_BUTTON_NAMES = [
    "Yes, get this app back up!",
    "Yes, get this app back up",
    "Get this app back up",
    "Wake up",
]
READY_TEXT = "Portfolio Decision"


def click_wake_button(page) -> bool:
    for name in WAKE_BUTTON_NAMES:
        button = page.get_by_role("button", name=name)
        try:
            button.wait_for(state="visible", timeout=5_000)
            button.click()
            print(f"Clicked wake button: {name}")
            return True
        except PlaywrightTimeoutError:
            continue
    return False


def wait_until_ready(page) -> None:
    page.get_by_text(READY_TEXT, exact=False).first.wait_for(
        timeout=180_000
    )


def main() -> int:
    screenshot_path = Path("wake-streamlit.png")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(30_000)

        print(f"Opening {APP_URL}")
        page.goto(APP_URL, wait_until="domcontentloaded")

        woke = click_wake_button(page)
        if not woke:
            print("No wake button found; checking whether the app is already live.")

        try:
            wait_until_ready(page)
        except PlaywrightTimeoutError:
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(
                "Timed out waiting for the dashboard. "
                f"Saved screenshot to {screenshot_path}"
            )
            print(f"Page title: {page.title()}")
            browser.close()
            return 1

        print("App is live.")
        browser.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
