from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_built_web_app_serves_benchmark_empty_answer_filter() -> None:
    dist_dir = Path("app/web/dist")
    assert (dist_dir / "index.html").exists()

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == HTTPStatus.OK
    assert "text/html" in response.headers["content-type"]

    emitted_javascript = "\n".join(
        path.read_text(encoding="utf-8") for path in (dist_dir / "assets").glob("*.js")
    )
    assert "Hide rows with an empty answer" in emitted_javascript
    assert "hide-empty-benchmark-answers" in emitted_javascript
