# tests/test_vector_store.py
import json
import tempfile
import os
import pytest
import backend.vector_store as vs


def test_vector_store_meta_loading():
    # Simulate a realm
    realm_id = "testrealm"

    # Get the metadata path for this realm
    _, meta_path = vs._get_paths(realm_id)

    # Make sure the directory exists
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    # Write fake metadata file
    data_lines = [
        json.dumps({"id": 1, "text": "foo"}) + "\n",
        json.dumps({"id": 2, "text": "bar"}) + "\n",
        json.dumps({"id": 3, "text": "baz"}) + "\n",
    ]
    with open(meta_path, "w") as f:
        f.writelines(data_lines)

    # Fake FAISS search results
    fake_I = [[0, 2]]  # indices from metadata
    fake_D = [[0.123, 0.456]]

    # Simulate reading metadata like in search()
    metas = []
    with open(meta_path) as f:
        lines = f.readlines()
    for idx in fake_I[0]:
        if idx < len(lines):
            metas.append(json.loads(lines[idx]))

    # Assertions
    assert fake_D[0] == [0.123, 0.456]
    assert metas == [
        {"id": 1, "text": "foo"},
        {"id": 3, "text": "baz"},
    ]
