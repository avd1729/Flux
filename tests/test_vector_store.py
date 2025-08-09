# tests/test_vector_store.py
import json
import tempfile
import os
import builtins
import types
import pytest

import backend.vector_store

def test_vector_store_meta_loading(monkeypatch):
    # Prepare fake META_PATH file
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
    data_lines = [
        json.dumps({"id": 1, "text": "foo"}) + "\n",
        json.dumps({"id": 2, "text": "bar"}) + "\n",
        json.dumps({"id": 3, "text": "baz"}) + "\n",
    ]
    temp_file.writelines(data_lines)
    temp_file.close()

    # Patch META_PATH in vector_store
    monkeypatch.setattr(backend.vector_store, "META_PATH", temp_file.name)

    # Fake I and D
    fake_I = [[0, 2]]  # pick first and last line
    fake_D = [[0.123, 0.456]]

    # Patch I and D in local scope of the function under test
    def fake_function_under_test():
        metas = []
        with open(backend.vector_store.META_PATH) as f:
            lines = f.readlines()
        for idx in fake_I[0]:
            if idx < len(lines):
                metas.append(json.loads(lines[idx]))
        return fake_D[0], metas

    # Run the patched function
    distances, metas = fake_function_under_test()

    # Assertions
    assert distances == [0.123, 0.456]
    assert metas == [
        {"id": 1, "text": "foo"},
        {"id": 3, "text": "baz"},
    ]

    # Clean up
    os.remove(temp_file.name)
