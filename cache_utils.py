"""
══════════════════════════════════════════════════════════════════════════
  SHARED CACHE UTILITY — v1.0
  Disk-persistent, TTL-based caching for all repetitive tasks
  Used by: screener, historical_validation, backtest_engine
══════════════════════════════════════════════════════════════════════════

WHAT THIS DOES:
  1. Caches yfinance downloads (price data, fundamentals) to disk
  2. Caches screener results so re-runs are instant
  3. Caches backtest results so re-runs are instant
  4. Caches NSE universe list so re-runs are instant
  5. All caches have TTL (time-to-live) so stale data expires

CACHE TYPES:
  - price_data: Historical OHLCV data from yfinance
  - fundamentals: Stock fundamentals from yfinance .info
  - screener_results: Full screener output (swing + core books)
  - backtest_results: Backtest trade logs and metrics
  - nse_universe: Nifty 500 symbol list
  - nifty_history: Nifty 50 index history for regime classification

USAGE:
  from cache_utils import get_cache, set_cache, clear_cache, cache_key

  # Check cache first
  data = get_cache("price_data", "RELIANCE", ttl_days=7)
  if data is None:
      data = yf.download(...)
      set_cache("price_data", "RELIANCE", data, ttl_days=7)

  # For screener results
  results = get_cache("screener_results", "latest", ttl_days=1)
  if results is None:
      results = run_screener()
      set_cache("screener_results", "latest", results, ttl_days=1)
"""

import os
import json
import time
import hashlib
import logging
from typing import Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("cache_utils")

# ═══════════════════════════════════════════════════════════════════════════
# CACHE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_DIR = ".cache"
_CACHE_VERSION = "v1"

# TTLs (time-to-live) for different cache types
_DEFAULT_TTLS = {
    "price_data": 7,        # 7 days — price data changes daily but history is stable
    "fundamentals": 1,      # 1 day — fundamentals change quarterly
    "screener_results": 1,  # 1 day — screener results are time-sensitive
    "backtest_results": 30, # 30 days — backtest results don't change unless logic changes
    "nse_universe": 7,      # 7 days — Nifty 500 list changes rarely
    "nifty_history": 1,     # 1 day — Nifty history updates daily
    "validation_snapshot": 30,  # 30 days — validation snapshots are expensive to compute
}

# Ensure cache directory exists
os.makedirs(_CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _cache_path(cache_type: str, key: str) -> str:
    """Generate cache file path."""
    # Sanitize key for filesystem
    safe_key = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{cache_type}_{safe_key}.json")


def _cache_meta_path(cache_type: str, key: str) -> str:
    """Generate cache metadata file path."""
    safe_key = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{cache_type}_{safe_key}_meta.json")


def _now_ts() -> float:
    """Current timestamp."""
    return time.time()


def _is_expired(created_ts: float, ttl_days: int) -> bool:
    """Check if cache entry is expired."""
    expiry = created_ts + (ttl_days * 24 * 3600)
    return time.time() > expiry


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON storage."""
    if isinstance(value, pd.DataFrame):
        return {"__type__": "dataframe", "data": value.to_dict(orient="split"), "index_type": str(type(value.index).__name__)}
    elif isinstance(value, pd.Series):
        return {"__type__": "series", "data": value.to_dict(), "index_type": str(type(value.index).__name__)}
    elif isinstance(value, datetime):
        return {"__type__": "datetime", "data": value.isoformat()}
    elif isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "data": value.tolist(), "dtype": str(value.dtype)}
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, tuple):
        return {"__type__": "tuple", "data": [_serialize_value(v) for v in value]}
    return value


def _deserialize_value(value: Any) -> Any:
    """Deserialize a value from JSON storage."""
    if isinstance(value, dict):
        if value.get("__type__") == "dataframe":
            import pandas as pd
            df = pd.DataFrame.from_dict(value["data"], orient="split")
            return df
        elif value.get("__type__") == "series":
            import pandas as pd
            return pd.Series(value["data"])
        elif value.get("__type__") == "datetime":
            return datetime.fromisoformat(value["data"])
        elif value.get("__type__") == "ndarray":
            import numpy as np
            return np.array(value["data"], dtype=value.get("dtype", "float64"))
        elif value.get("__type__") == "tuple":
            return tuple(_deserialize_value(v) for v in value["data"])
        else:
            return {k: _deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_deserialize_value(v) for v in value]
    return value


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC CACHE API
# ═══════════════════════════════════════════════════════════════════════════

def get_cache(cache_type: str, key: str, ttl_days: Optional[int] = None) -> Optional[Any]:
    """
    Get a value from cache.

    Args:
        cache_type: Type of cache (price_data, fundamentals, etc.)
        key: Unique key for this cache entry
        ttl_days: Override default TTL for this cache type

    Returns:
        Cached value or None if not found/expired
    """
    if ttl_days is None:
        ttl_days = _DEFAULT_TTLS.get(cache_type, 7)

    path = _cache_path(cache_type, key)
    meta_path = _cache_meta_path(cache_type, key)

    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None

    try:
        # Read metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        created_ts = meta.get("created_ts", 0)
        if _is_expired(created_ts, ttl_days):
            # Expired — clean up
            _delete_cache(cache_type, key)
            return None

        # Read data
        with open(path, "r") as f:
            raw = json.load(f)

        return _deserialize_value(raw)

    except Exception as e:
        logger.debug(f"Cache read error for {cache_type}:{key}: {e}")
        return None


def set_cache(cache_type: str, key: str, value: Any, ttl_days: Optional[int] = None) -> bool:
    """
    Set a value in cache.

    Args:
        cache_type: Type of cache
        key: Unique key for this cache entry
        value: Value to cache
        ttl_days: Override default TTL for this cache type

    Returns:
        True if successful, False otherwise
    """
    if ttl_days is None:
        ttl_days = _DEFAULT_TTLS.get(cache_type, 7)

    path = _cache_path(cache_type, key)
    meta_path = _cache_meta_path(cache_type, key)

    try:
        # Write data
        serialized = _serialize_value(value)
        with open(path, "w") as f:
            json.dump(serialized, f, default=str)

        # Write metadata
        meta = {
            "created_ts": _now_ts(),
            "ttl_days": ttl_days,
            "cache_type": cache_type,
            "key": key,
            "version": _CACHE_VERSION,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        return True

    except Exception as e:
        logger.debug(f"Cache write error for {cache_type}:{key}: {e}")
        return False


def _delete_cache(cache_type: str, key: str) -> None:
    """Delete a cache entry."""
    path = _cache_path(cache_type, key)
    meta_path = _cache_meta_path(cache_type, key)
    for p in [path, meta_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def clear_cache(cache_type: Optional[str] = None, older_than_days: Optional[int] = None) -> int:
    """
    Clear cache entries.

    Args:
        cache_type: If provided, only clear this cache type. If None, clear all.
        older_than_days: If provided, only clear entries older than this many days.

    Returns:
        Number of entries cleared
    """
    cleared = 0
    now = _now_ts()

    for filename in os.listdir(_CACHE_DIR):
        if not filename.endswith("_meta.json"):
            continue

        # Extract cache_type from filename
        parts = filename.replace("_meta.json", "").rsplit("_", 1)
        if len(parts) != 2:
            continue

        entry_cache_type = parts[0]

        if cache_type is not None and entry_cache_type != cache_type:
            continue

        meta_path = os.path.join(_CACHE_DIR, filename)
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            created_ts = meta.get("created_ts", 0)
            ttl_days = meta.get("ttl_days", 7)

            # Check if expired by TTL
            if _is_expired(created_ts, ttl_days):
                key = meta.get("key", "")
                _delete_cache(entry_cache_type, key)
                cleared += 1
                continue

            # Check if older than specified days
            if older_than_days is not None:
                cutoff = now - (older_than_days * 24 * 3600)
                if created_ts < cutoff:
                    key = meta.get("key", "")
                    _delete_cache(entry_cache_type, key)
                    cleared += 1

        except Exception:
            pass

    return cleared


def clear_all_cache() -> int:
    """Clear all cache entries. Returns number cleared."""
    return clear_cache()


def get_cache_stats() -> dict:
    """Get statistics about the cache."""
    stats = {}
    total_size = 0
    total_entries = 0

    for filename in os.listdir(_CACHE_DIR):
        if not filename.endswith("_meta.json"):
            continue

        parts = filename.replace("_meta.json", "").rsplit("_", 1)
        if len(parts) != 2:
            continue

        entry_cache_type = parts[0]
        meta_path = os.path.join(_CACHE_DIR, filename)

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            data_path = _cache_path(entry_cache_type, meta.get("key", ""))
            if os.path.exists(data_path):
                size = os.path.getsize(data_path)
                total_size += size

            if entry_cache_type not in stats:
                stats[entry_cache_type] = {"count": 0, "size_bytes": 0}
            stats[entry_cache_type]["count"] += 1
            stats[entry_cache_type]["size_bytes"] += size
            total_entries += 1

        except Exception:
            pass

    return {
        "total_entries": total_entries,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "by_type": stats,
        "cache_dir": os.path.abspath(_CACHE_DIR),
    }


def print_cache_stats() -> None:
    """Print cache statistics."""
    stats = get_cache_stats()
    print(f"\n{'═' * 60}")
    print("  📦 CACHE STATISTICS")
    print(f"{'═' * 60}")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total size: {stats['total_size_mb']} MB")
    print(f"  Cache dir: {stats['cache_dir']}")
    if stats["by_type"]:
        print(f"\n  By type:")
        for cache_type, info in stats["by_type"].items():
            print(f"    {cache_type}: {info['count']} entries, {info['size_bytes'] / 1024:.1f} KB")
    print(f"{'═' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# SPECIALIZED CACHE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def cache_price_data(symbol: str, df: Any, ttl_days: int = 7) -> bool:
    """Cache price data for a symbol."""
    return set_cache("price_data", symbol, df, ttl_days)


def get_price_data(symbol: str, ttl_days: int = 7) -> Optional[Any]:
    """Get cached price data for a symbol."""
    return get_cache("price_data", symbol, ttl_days)


def cache_fundamentals(symbol: str, fundamentals: Any, ttl_days: int = 1) -> bool:
    """Cache fundamentals for a symbol."""
    return set_cache("fundamentals", symbol, fundamentals, ttl_days)


def get_fundamentals(symbol: str, ttl_days: int = 1) -> Optional[Any]:
    """Get cached fundamentals for a symbol."""
    return get_cache("fundamentals", symbol, ttl_days)


def cache_screener_results(results: Any, key: str = "latest", ttl_days: int = 1) -> bool:
    """Cache screener results."""
    return set_cache("screener_results", key, results, ttl_days)


def get_screener_results(key: str = "latest", ttl_days: int = 1) -> Optional[Any]:
    """Get cached screener results."""
    return get_cache("screener_results", key, ttl_days)


def cache_backtest_results(results: Any, key: str = "latest", ttl_days: int = 30) -> bool:
    """Cache backtest results."""
    return set_cache("backtest_results", key, results, ttl_days)


def get_backtest_results(key: str = "latest", ttl_days: int = 30) -> Optional[Any]:
    """Get cached backtest results."""
    return get_cache("backtest_results", key, ttl_days)


def cache_nse_universe(symbols: list, ttl_days: int = 7) -> bool:
    """Cache NSE universe list."""
    return set_cache("nse_universe", "nifty500", symbols, ttl_days)


def get_nse_universe(ttl_days: int = 7) -> Optional[list]:
    """Get cached NSE universe list."""
    return get_cache("nse_universe", "nifty500", ttl_days)


def cache_nifty_history(df: Any, ttl_days: int = 1) -> bool:
    """Cache Nifty history data."""
    return set_cache("nifty_history", "nifty50", df, ttl_days)


def get_nifty_history(ttl_days: int = 1) -> Optional[Any]:
    """Get cached Nifty history data."""
    return get_cache("nifty_history", "nifty50", ttl_days)


def cache_validation_snapshot(snapshot_date: str, results: Any, ttl_days: int = 30) -> bool:
    """Cache validation snapshot results."""
    return set_cache("validation_snapshot", snapshot_date, results, ttl_days)


def get_validation_snapshot(snapshot_date: str, ttl_days: int = 30) -> Optional[Any]:
    """Get cached validation snapshot results."""
    return get_cache("validation_snapshot", snapshot_date, ttl_days)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE KEY GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def price_data_key(symbol: str, start: str, end: str) -> str:
    """Generate cache key for price data."""
    return f"price_{symbol}_{start}_{end}"


def fundamentals_key(symbol: str) -> str:
    """Generate cache key for fundamentals."""
    return f"fund_{symbol}"


def screener_results_key(universe_hash: str, timestamp: str) -> str:
    """Generate cache key for screener results."""
    return f"screener_{universe_hash}_{timestamp}"


def backtest_results_key(symbols_hash: str, start: str, end: str, mode: str) -> str:
    """Generate cache key for backtest results."""
    return f"backtest_{symbols_hash}_{start}_{end}_{mode}"


def validation_snapshot_key(snapshot_date: str, universe_hash: str) -> str:
    """Generate cache key for validation snapshot."""
    return f"validation_{snapshot_date}_{universe_hash}"


# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def init_cache(clear_expired: bool = True) -> dict:
    """
    Initialize cache system.

    Args:
        clear_expired: If True, clear expired entries on init

    Returns:
        Cache statistics
    """
    if clear_expired:
        cleared = clear_cache()
        if cleared > 0:
            logger.info(f"Cache init: cleared {cleared} expired entries")

    stats = get_cache_stats()
    logger.info(f"Cache init: {stats['total_entries']} entries, {stats['total_size_mb']} MB")
    return stats


# Auto-init on import
init_cache()
