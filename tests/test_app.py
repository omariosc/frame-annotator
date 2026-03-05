import json
import os
import tempfile
from pathlib import Path

import pytest
from frame_annotator.app import create_app
from frame_annotator.config import get_default_config


@pytest.fixture
def app_with_frames(tmp_path):
    # Create some dummy frames
    for i in range(5):
        (tmp_path / f"frame_{i:04d}.png").write_bytes(b"fake png")
    config = get_default_config()
    app = create_app(str(tmp_path), config)
    app.config["TESTING"] = True
    return app


def test_index(app_with_frames):
    with app_with_frames.test_client() as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Frame Annotator" in resp.data


def test_get_frames(app_with_frames):
    with app_with_frames.test_client() as client:
        resp = client.get("/api/frames")
        data = json.loads(resp.data)
        assert data["total"] == 5
        assert len(data["frames"]) == 5


def test_load_annotations_empty(app_with_frames):
    with app_with_frames.test_client() as client:
        resp = client.get("/api/load_annotations")
        data = json.loads(resp.data)
        assert data["clips"] == []


def test_save_and_load_annotations(app_with_frames):
    with app_with_frames.test_client() as client:
        payload = {"clips": [{"start": 0, "end": 2, "class": "positive"}]}
        resp = client.post("/api/save_annotations",
                          data=json.dumps(payload),
                          content_type="application/json")
        data = json.loads(resp.data)
        assert data["success"] is True

        resp = client.get("/api/load_annotations")
        data = json.loads(resp.data)
        assert len(data["clips"]) == 1
        assert data["clips"][0]["class"] == "positive"
