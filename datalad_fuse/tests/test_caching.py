from __future__ import annotations

from pathlib import Path
import subprocess

from datalad.api import Dataset
from fsspec.caching import caches
import pytest

from datalad_fuse.caching import COALESCING_BLOCK_CACHE, CoalescingBlockCache
from datalad_fuse.fsspec import DatasetAdapter, FsspecAdapter


def test_coalescing_block_cache() -> None:
    assert caches[COALESCING_BLOCK_CACHE] is CoalescingBlockCache
    content = bytes(range(256)) * 40
    requests: list[tuple[int, int]] = []

    def fetcher(start: int, end: int) -> bytes:
        requests.append((start, end))
        return content[start:end]

    c = CoalescingBlockCache(100, fetcher, len(content), maxblocks=5)
    # The blocks missing for a read are fetched with one request
    assert c._fetch(150, 420) == content[150:420]
    assert requests == [(100, 500)]
    # Cached blocks are not fetched again, and runs of missing blocks around
    # them are fetched with one request each
    assert c._fetch(30, 720) == content[30:720]
    assert requests[1:] == [(0, 100), (500, 800)]
    # Reads at the end of the content, and beyond
    assert c._fetch(10_190, 20_000) == content[10_190:]
    assert requests[3:] == [(10_100, 10_240)]
    assert c._fetch(20_000, 20_100) == b""
    # Only the 5 most recently used blocks are kept (but at least the blocks
    # of the last read)
    assert sorted(c.blocks) == [5, 6, 7, 101, 102]
    requests.clear()
    assert c._fetch(0, 10_240) == content
    assert requests == [(0, 500), (800, 10_100)]
    assert len(c.blocks) == 103


@pytest.fixture
def annex_clone(tmp_path: Path) -> Path:
    """A ``git clone`` (with git-annex not initialized) of a dataset with an
    annexed file, whose content is thus missing"""
    ds = Dataset(tmp_path / "ds").create()
    (ds.pathobj / "data.bin").write_bytes(b"0123456789" * 1000)
    ds.save(message="Add data")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(ds.path), str(clone)], check=True)
    return clone


def annex_uuid(path: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(path), "config", "--get", "annex.uuid"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


@pytest.mark.usefixtures("tmp_home")
def test_adapter_no_init_annex(annex_clone: Path) -> None:
    dsap = DatasetAdapter(annex_clone, caching=False, init_annex=False)
    try:
        assert not dsap.annex_initialized
        assert dsap.annex_disabled
        state, key = dsap.get_file_state("data.bin")
        assert state.name == "NO_CONTENT"
        assert key is not None and key.size == 10_000
        with pytest.raises(IOError, match="not initialized"):
            dsap.open("data.bin")
    finally:
        dsap.close()
    assert annex_uuid(annex_clone) == ""


@pytest.mark.usefixtures("tmp_home")
def test_adapter_fs_options(annex_clone: Path) -> None:
    with FsspecAdapter(
        annex_clone,
        caching=False,
        init_annex=False,
        fsspec_options={
            "block_size": 1024,
            "cache_type": COALESCING_BLOCK_CACHE,
            "cache_options": {"maxblocks": 7},
        },
    ) as adapter:
        dsap, relpath = adapter.resolve_dataset(annex_clone / "data.bin")
        assert relpath == "data.bin"
        assert dsap.fs.block_size == 1024
        assert dsap.fs.cache_type == COALESCING_BLOCK_CACHE
        assert dsap.fs.cache_options == {"maxblocks": 7}
