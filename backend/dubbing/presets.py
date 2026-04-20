"""Presets — named pipeline configurations with save/load/list.

Up to 8 user-created presets stored as JSON files.
Each preset captures the full DubbingSettings snapshot so the user
can switch between configurations with one click (tab).

Storage: {work_root}/presets/{slug}.json
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

MAX_PRESETS = 8
PRESETS_DIR_NAME = "presets"


def _presets_dir(work_root: Path) -> Path:
    d = work_root / PRESETS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slugify(name: str) -> str:
    """Convert preset name to safe filename slug."""
    slug = re.sub(r'[^\w\s-]', '', name.lower().strip())
    slug = re.sub(r'[\s_-]+', '-', slug)
    return slug[:60] or "preset"


# ─── CRUD ───────────────────────────────────────────────────────────────

def save_preset(
    work_root: Path,
    name: str,
    settings: dict,
) -> dict:
    """Save a named preset. Enforces MAX_PRESETS limit.

    Returns the saved preset metadata.
    Raises ValueError if limit reached and name is new.
    """
    slug = _slugify(name)
    presets_dir = _presets_dir(work_root)

    # Check limit and slug collision
    existing = list_presets(work_root)
    existing_map = {p["slug"]: p["name"] for p in existing}
    if slug not in existing_map and len(existing) >= MAX_PRESETS:
        raise ValueError(
            f"Maximum {MAX_PRESETS} presets allowed. "
            f"Delete one before creating a new one."
        )
    # Warn if slug exists under a different display name (collision)
    if slug in existing_map and existing_map[slug] != name:
        log.info("Preset slug '%s' exists as '%s', overwriting with '%s'",
                 slug, existing_map[slug], name)

    preset_data = {
        "name": name,
        "slug": slug,
        "settings": settings,
    }

    path = presets_dir / f"{slug}.json"
    path.write_text(json.dumps(preset_data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Preset saved: %s → %s", name, path)
    return preset_data


def _safe_slug(slug: str) -> str:
    """Sanitize slug to prevent path traversal."""
    # Use the same slugify logic to ensure only safe characters
    clean = _slugify(slug)
    if not clean:
        raise ValueError("Invalid preset slug")
    return clean


def load_preset(work_root: Path, slug: str) -> Optional[dict]:
    """Load a preset by slug. Returns None if not found."""
    slug = _safe_slug(slug)
    path = _presets_dir(work_root) / f"{slug}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Validate required keys
        if not isinstance(data, dict) or "settings" not in data or not isinstance(data["settings"], dict):
            log.warning("Preset %s is corrupted (missing 'settings' dict)", slug)
            return None
        data.setdefault("name", slug)
        data.setdefault("slug", slug)
        return data
    except Exception as e:
        log.warning("Failed to load preset %s: %s", slug, e)
        return None


def list_presets(work_root: Path) -> List[dict]:
    """List all saved presets (name + slug, no full settings)."""
    presets_dir = _presets_dir(work_root)
    result = []
    for path in sorted(presets_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result.append({
                "name": data.get("name", path.stem),
                "slug": data.get("slug", path.stem),
            })
        except Exception:
            continue
    return result


def delete_preset(work_root: Path, slug: str) -> bool:
    """Delete a preset by slug. Returns True if deleted."""
    slug = _safe_slug(slug)
    path = _presets_dir(work_root) / f"{slug}.json"
    if path.exists():
        path.unlink()
        log.info("Preset deleted: %s", slug)
        return True
    return False
