import json

import numpy as np

from anchor.embed.cache import cache_key, cached_embed


def test_second_call_hits_cache(bow, tmp_path):
    texts = ["apple banana", "cherry"]
    first = cached_embed(bow, texts, tmp_path)
    second = cached_embed(bow, texts, tmp_path)
    assert bow.calls == 1
    np.testing.assert_array_equal(first, second)
    assert second.dtype == np.float32 and second.shape == (2, bow.dim)


def test_any_change_misses_cache(bow, tmp_path):
    cached_embed(bow, ["apple", "cherry"], tmp_path)
    cached_embed(bow, ["apple", "cherry date"], tmp_path)  # one text edited
    cached_embed(bow, ["cherry", "apple"], tmp_path)  # same texts, new order
    assert bow.calls == 3


def test_model_name_is_part_of_key(bow, tmp_path):
    cached_embed(bow, ["apple"], tmp_path)
    bow.name = "bow-v2"
    cached_embed(bow, ["apple"], tmp_path)
    assert bow.calls == 2


def test_key_is_unambiguous():
    assert cache_key("m", ["ab", "c"]) != cache_key("m", ["a", "bc"])
    assert cache_key("m", ["x"]) != cache_key("mx", [])


def test_writes_metadata_and_no_temp_files(bow, tmp_path):
    cached_embed(bow, ["apple", "banana"], tmp_path)
    files = sorted(p.suffix for p in (tmp_path / "bow").iterdir())
    assert files == [".json", ".npy"]
    meta = json.loads(next((tmp_path / "bow").glob("*.json")).read_text())
    assert meta["model"] == "bow" and meta["count"] == 2 and meta["dim"] == bow.dim
