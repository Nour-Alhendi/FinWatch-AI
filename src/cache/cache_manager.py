"""
FinWatch AI — Pipeline Cache Manager
======================================
Cache key = sha256( input_files_hash | source_code_hash | upstream_layer_hash )

Any change in input data, source code, or an upstream layer invalidates the cache.
Cascade invalidation is automatic: each layer's combined_hash feeds into the next
layer's key, so a change upstream ripples forward without manual bookkeeping.

Cache entries live in data/cache/{layer_name}.json (gitignored via data/).
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT      = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"


class CacheManager:
    def __init__(self, force: bool = False, force_from_layer: Optional[int] = None):
        """
        force            — bypass ALL layer caches (--force flag)
        force_from_layer — bypass cache for this layer number and all higher ones (--force-layer N)
        """
        self.force            = force
        self.force_from_layer = force_from_layer
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Hashing helpers ──────────────────────────────────────────────────────

    def _hash_dirs(self, directories: list[Path]) -> str:
        """Hash input parquet files by name + size + mtime_ns (fast, reliable)."""
        parts = []
        for d in directories:
            if not d.exists():
                parts.append(f"{d}:missing")
                continue
            for f in sorted(d.glob("*.parquet")):
                st = f.stat()
                parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20]

    def _hash_sources(self, source_dirs: list[Path]) -> str:
        """Hash all .py file contents under each source directory (sorted)."""
        h = hashlib.sha256()
        for d in source_dirs:
            for f in sorted(d.rglob("*.py")):
                h.update(f.read_bytes())
        return h.hexdigest()[:20]

    def compute_hash(
        self,
        input_dirs:   list[Path],
        source_dirs:  list[Path],
        upstream_hash: str = "",
    ) -> str:
        """Return the combined cache key for a layer."""
        input_h    = self._hash_dirs(input_dirs)
        code_h     = self._hash_sources(source_dirs)
        combined   = f"{input_h}|{code_h}|{upstream_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:24]

    # ── Cache entry I/O ──────────────────────────────────────────────────────

    def _entry_path(self, layer_name: str) -> Path:
        return CACHE_DIR / f"{layer_name}.json"

    def get_stored_hash(self, layer_name: str) -> str:
        """Return the combined_hash stored for this layer (used as upstream_hash by the next layer)."""
        p = self._entry_path(layer_name)
        if not p.exists():
            return ""
        try:
            return json.loads(p.read_text()).get("combined_hash", "")
        except Exception:
            return ""

    def is_valid(self, layer_name: str, current_hash: str, layer_num: int) -> bool:
        """Return True if the cached result is still valid and should be used."""
        if self.force:
            return False
        if self.force_from_layer is not None and layer_num >= self.force_from_layer:
            return False
        stored = self.get_stored_hash(layer_name)
        return stored == current_hash

    def save(self, layer_name: str, combined_hash: str) -> None:
        """Persist the cache entry after a successful layer run."""
        entry = {
            "layer":         layer_name,
            "combined_hash": combined_hash,
            "computed_at":   datetime.now().isoformat(),
        }
        self._entry_path(layer_name).write_text(json.dumps(entry, indent=2))

    def invalidate(self, layer_name: str) -> None:
        """Explicitly remove a cache entry (e.g. after an upstream change)."""
        p = self._entry_path(layer_name)
        if p.exists():
            p.unlink()

    # ── Context-manager convenience ──────────────────────────────────────────

    def run_layer(
        self,
        layer_name:    str,
        layer_num:     int,
        input_dirs:    list[Path],
        source_dirs:   list[Path],
        upstream_hash: str,
        fn,
        label:         str = "",
    ) -> str:
        """
        Run fn() only when the cache is invalid.
        Returns the combined_hash for this layer (to pass as upstream_hash to the next).

        Prints clearly: '[cache] Layer X — loaded from cache' or '... — computed'.
        """
        tag   = label or layer_name
        h     = self.compute_hash(input_dirs, source_dirs, upstream_hash)
        valid = self.is_valid(layer_name, h, layer_num)

        if valid:
            print(f"\n  [cache] {tag} — loaded from cache  (hash: {h})")
            return h

        fn()
        # Recompute hash AFTER the layer wrote its outputs (mtime/size may have changed)
        h_after = self.compute_hash(input_dirs, source_dirs, upstream_hash)
        self.save(layer_name, h_after)
        print(f"\n  [cache] {tag} — computed and cached (hash: {h_after})")
        return h_after
