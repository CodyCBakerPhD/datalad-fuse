"""fsspec caches suited to the random access patterns of, e.g., HDF5 readers

.. note::
   `CoalescingBlockCache` only differs from fsspec's ``"blockcache"`` in
   fetching runs of consecutive missing blocks with one request.  Once
   fsspec's ``BlockCache`` does so too (proposed in
   https://github.com/CodyCBakerPhD/filesystem_spec/pull/1), this module can be
   dropped in favor of ``cache_type="blockcache"``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

from fsspec.caching import BaseCache, register_cache

#: Name under which `CoalescingBlockCache` is registered with fsspec, to be
#: passed as ``cache_type``
COALESCING_BLOCK_CACHE = "coalescing-blockcache"


class CoalescingBlockCache(BaseCache):
    """Keep the least recently used blocks of a file, fetching runs of
    consecutive missing blocks with one request each

    Like fsspec's ``"blockcache"``, which suits readers jumping over many
    small pieces scattered over a file (where the default read-ahead cache
    would discard what it read ahead at each jump), but a read spanning many
    missing blocks is fetched with one request per run of consecutive missing
    blocks rather than with one request per block.

    Parameters
    ----------
    blocksize : int
        Size of the blocks in bytes.
    fetcher : callable
        ``fetcher(start, end)`` returns the bytes of the file in that range.
    size : int
        Size of the file.
    maxblocks : int
        Maximum number of blocks kept (more are kept while a single read
        needs them).
    """

    name = COALESCING_BLOCK_CACHE

    def __init__(
        self,
        blocksize: int,
        fetcher: Callable[[int, int], bytes],
        size: int,
        maxblocks: int = 32,
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.maxblocks = maxblocks
        self.blocks: OrderedDict[int, bytes] = OrderedDict()

    def _fetch(self, start: Optional[int], end: Optional[int]) -> bytes:
        if start is None:
            start = 0
        if end is None or end > self.size:
            end = self.size
        if start >= end:
            return b""
        first = start // self.blocksize
        last = (end - 1) // self.blocksize
        numbers = range(first, last + 1)
        missing = [n for n in numbers if n not in self.blocks]
        i = 0
        while i < len(missing):
            j = i
            while j + 1 < len(missing) and missing[j + 1] == missing[j] + 1:
                j += 1
            run_start = missing[i] * self.blocksize
            run_end = min((missing[j] + 1) * self.blocksize, self.size)
            data = self.fetcher(run_start, run_end)
            for n in range(missing[i], missing[j] + 1):
                offset = (n - missing[i]) * self.blocksize
                self.blocks[n] = data[offset : offset + self.blocksize]
            i = j + 1
        out = []
        for n in numbers:
            self.blocks.move_to_end(n)
            out.append(self.blocks[n])
        # Evict the least recently used blocks, but not those just read
        while len(self.blocks) > max(self.maxblocks, len(numbers)):
            self.blocks.popitem(last=False)
        data = b"".join(out)
        offset = first * self.blocksize
        return data[start - offset : end - offset]


register_cache(CoalescingBlockCache, clobber=True)
