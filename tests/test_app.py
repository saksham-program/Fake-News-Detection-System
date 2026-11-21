import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # noqa: E402
from preprocessing import clean_text  # noqa: E402

REAL = "WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a budget bill after weeks of negotiations."
FAKE = "BREAKING!!! Shocking secret the media won't tell you, share before they delete this"


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_clean_text():
    assert clean_text("The <b>Cats</b> visited https://x.com!") == "cat visited"
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


@pytest.mark.parametrize("path", ["/", "/predict", "/about", "/readme", "/health"])
def test_pages(client, path):
    assert client.get(path).status_code == 200


def test_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_api_predict(client):
    for text in (REAL, FAKE):
        resp = client.post("/api/predict", json={"text": text})
        body = resp.get_json()
        assert resp.status_code == 200 and body["success"]
        assert body["result"]["label"] in {"Real News", "Fake News"}
        assert abs(body["result"]["real_probability"] + body["result"]["fake_probability"] - 1) < 1e-3


def test_empty_input(client):
    assert client.post("/api/predict", json={"text": ""}).status_code == 400


def test_txt_upload_and_bad_extension(client):
    ok = client.post("/predict", data={"news_file": (io.BytesIO(REAL.encode()), "a.txt")}, content_type="multipart/form-data")
    assert ok.status_code == 200
    bad = client.post("/predict", data={"news_file": (io.BytesIO(b"x"), "a.exe")}, content_type="multipart/form-data")
    assert bad.status_code == 400
