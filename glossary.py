"""Glossary support for meeting minutes generation.

This module loads glossary definitions from YAML files and prepares
a compact, human-readable rules block for inclusion into the LLM prompt.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Dict, Tuple, Optional

import yaml

from meeting_summary_prompts import GLOSSARY_SECTION_TITLE, GLOSSARY_SECTION_INTRO

logger = logging.getLogger(__name__)


@dataclass
class GlossaryEntry:
    """Single glossary entry with canonical term and its variants."""

    canonical: str
    variants: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def normalized_variants(self) -> List[str]:
        """Return variants cleaned from empty strings and duplicates."""
        seen = set()
        result: List[str] = []
        for v in self.variants:
            v_clean = (v or "").strip()
            if not v_clean:
                continue
            if v_clean.lower() in seen:
                continue
            seen.add(v_clean.lower())
            result.append(v_clean)
        return result


class Glossary:
    """Container for glossary entries plus helpers for prompt generation."""

    def __init__(self, entries: Optional[Iterable[GlossaryEntry]] = None) -> None:
        self.entries: List[GlossaryEntry] = list(entries or [])

    def is_empty(self) -> bool:
        """Return True if no usable entries exist."""
        return not any(e.normalized_variants() for e in self.entries)

    def clean_protocol_text(self, text: str) -> str:
        """Remove incorrect variant mentions in parentheses after canonical terms.

        Example: "JWT (GVT)" -> "JWT"
        This is a lightweight post-processing step to remove cases where LLM
        helpfully adds the incorrect variant in parentheses despite instructions.
        """
        if not text or self.is_empty():
            return text

        # Build a map of canonical -> set of variants (normalized to lowercase)
        canonical_to_variants: Dict[str, set] = {}
        for entry in self.entries:
            variants_lower = {v.lower() for v in entry.normalized_variants()}
            canonical_to_variants[entry.canonical] = variants_lower

        # For each canonical term, remove patterns like "Canonical (variant)"
        for canonical, variants_set in canonical_to_variants.items():
            # Escape canonical for regex
            canonical_escaped = re.escape(canonical)
            # Build a pattern that matches any variant in parentheses
            # We want to match: "Canonical (variant1)" or "Canonical (variant2)"
            for variant in variants_set:
                variant_escaped = re.escape(variant)
                # Case-insensitive pattern: canonical followed by optional spaces, open paren, variant, close paren
                pattern = rf"\b{canonical_escaped}\s*\(\s*{variant_escaped}\s*\)"
                # Replace with just the canonical term
                text = re.sub(pattern, canonical, text, flags=re.IGNORECASE)

        return text

    def force_replace_variants(self, text: str) -> str:
        """Force replace all standalone variant occurrences with canonical terms.

        This is an aggressive post-processing step that replaces any standalone
        word matching a glossary variant (case-insensitive) with its canonical term.

        Example: "GVT токен" -> "JWT токен", "Кавка брокер" -> "Kafka брокер"
        """
        if not text or self.is_empty():
            return text

        # Build a map: variant (lowercase) -> canonical term
        variant_to_canonical: Dict[str, str] = {}
        for entry in self.entries:
            for variant in entry.normalized_variants():
                variant_to_canonical[variant.lower()] = entry.canonical

        # Sort variants by length (descending) to handle multi-word variants first
        sorted_variants = sorted(variant_to_canonical.keys(), key=len, reverse=True)

        # Replace each variant with its canonical term
        for variant_lower in sorted_variants:
            canonical = variant_to_canonical[variant_lower]
            # Escape variant for regex
            variant_escaped = re.escape(variant_lower)
            # Match variant as a whole word (case-insensitive)
            pattern = rf"\b{variant_escaped}\b"
            # Replace with canonical term, preserving surrounding text
            text = re.sub(pattern, canonical, text, flags=re.IGNORECASE)

        return text

    def to_prompt_block(self) -> str:
        """Render glossary rules as a text block for inclusion into system prompt.

        The block is written in Russian and explains how to map variants to
        canonical terms when composing the meeting protocol.
        """
        if self.is_empty():
            return ""

        lines: List[str] = []
        lines.append(f"{GLOSSARY_SECTION_TITLE}\n")
        lines.append(GLOSSARY_SECTION_INTRO)

        # Sort for stable, predictable order
        for entry in sorted(self.entries, key=lambda e: e.canonical.lower()):
            variants = entry.normalized_variants()
            if not variants:
                continue

            variants_str = ", ".join(f'"{v}"' for v in variants)
            line = (
                f'- Если в тексте встречаются варианты: {variants_str}, '
                f'используй в протоколе термин: "{entry.canonical}".'
            )
            if entry.description:
                line += f" {entry.description}"
            lines.append(line)

        return "\n".join(lines)


def _load_yaml(path: str) -> List[dict]:
    """Load raw YAML data from file, returning a list of entry dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Glossary file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []

    if not isinstance(data, list):
        raise ValueError(f"Glossary YAML must contain a list of entries, got {type(data)!r}")

    return data


def load_glossary(path: str) -> Glossary:
    """Load a single glossary from a YAML file."""
    raw_entries = _load_yaml(path)
    entries: List[GlossaryEntry] = []

    for idx, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            logger.warning("Skipping non-dict glossary entry at index %s in %s", idx, path)
            continue

        canonical = (raw.get("canonical") or "").strip()
        variants_raw = raw.get("variants") or []
        description_raw = raw.get("description")

        if not canonical:
            logger.warning("Skipping glossary entry without canonical term at index %s in %s", idx, path)
            continue

        if not isinstance(variants_raw, list):
            logger.warning(
                "Skipping glossary entry with non-list variants at index %s in %s", idx, path
            )
            continue

        variants = []
        for v in variants_raw:
            v_str = (str(v) if v is not None else "").strip()
            if v_str:
                variants.append(v_str)

        if not variants:
            logger.warning(
                "Skipping glossary entry without usable variants at index %s in %s", idx, path
            )
            continue

        description = None
        if description_raw is not None:
            description_str = str(description_raw).strip()
            description = description_str if description_str else None

        entries.append(
            GlossaryEntry(
                canonical=canonical,
                variants=variants,
                description=description,
            )
        )

    if not entries:
        logger.info("Loaded glossary from %s, but no valid entries were found", path)
    else:
        logger.info("Loaded %d glossary entries from %s", len(entries), path)

    return Glossary(entries)


def merge_glossaries(base: Glossary, extra: Glossary) -> Glossary:
    """Merge base and extra glossaries with extra taking precedence on conflicts.

    Conflict rule:
    - If the same variant appears in both glossaries but points to different
      canonical terms, the canonical from the extra glossary is used and
      a warning is logged.
    """
    # Map normalized variant -> (canonical, description, source, surface_form)
    variant_map: Dict[str, Tuple[str, Optional[str], str, str]] = {}

    def add_entries(entries: Iterable[GlossaryEntry], source: str, overwrite: bool) -> None:
        for entry in entries:
            for v in entry.normalized_variants():
                key = v.lower()
                if key in variant_map and not overwrite:
                    # Base glossary keeps existing mapping
                    continue

                if key in variant_map and overwrite:
                    prev_canonical, _, prev_source, _ = variant_map[key]
                    if prev_canonical != entry.canonical:
                        logger.warning(
                            "Glossary conflict for variant %r: %r from %s overridden by %r from %s",
                            v,
                            prev_canonical,
                            prev_source,
                            entry.canonical,
                            source,
                        )

                variant_map[key] = (entry.canonical, entry.description, source, v)

    # Base glossary first, then extra with overwrite
    add_entries(base.entries, source="base", overwrite=False)
    add_entries(extra.entries, source="extra", overwrite=True)

    # Rebuild entries grouped by canonical term
    canonical_map: Dict[str, Tuple[set, Optional[str]]] = {}
    for _, (canonical, description, _, surface_form) in variant_map.items():
        if canonical not in canonical_map:
            canonical_map[canonical] = (set(), description)
        variants_set, existing_description = canonical_map[canonical]
        variants_set.add(surface_form)
        # Prefer any non-empty description, later extra entries may override
        if description:
            canonical_map[canonical] = (variants_set, description)
        else:
            canonical_map[canonical] = (variants_set, existing_description)

    merged_entries: List[GlossaryEntry] = []
    for canonical, (variants_set, description) in canonical_map.items():
        merged_entries.append(
            GlossaryEntry(
                canonical=canonical,
                variants=sorted(variants_set),
                description=description,
            )
        )

    logger.info(
        "Merged glossaries: %d base entries + %d extra entries -> %d merged entries",
        len(base.entries),
        len(extra.entries),
        len(merged_entries),
    )

    return Glossary(merged_entries)


