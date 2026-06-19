"""Structure filter (S8) — chapters and subchapters in ONE pass.

Single responsibility (§4.2 Rag_pipeline_architecture.md): cut the flat text from
`01_extract/full_text.txt` into a section tree (chapter = depth 1, subchapter = depth 2) and
write `02_structure/sections.jsonl` — one valid `Section` per line
(the Structure→Chunk pipe from §3 / contracts.py). Implements the orchestrator step contract (S6).

The boundary signals are tuned against the REAL pdfminer.six output (Designing_Data.pdf), NOT
the idealized format of the old doc:
  • chapter  = the running-header line `Chapter N: <Title>` (repeated on every
    page of the chapter) → reliably gives BOTH the number AND the title; the boundary = where the number changes;
  • subchapters = the BOTTOM footers `<Section Name> | <pagenum>`, which pdfminer splits into
    THREE non-empty lines (name / `|` / number). Reconstructing the triple gives a reliable list
    of section names; each name is searched in the body as a clean heading line → cut point.

DI / config-agnostic: `external_id` and (optionally) the chapter pattern come via the constructor.
Markers (running headers, footer fragments) do NOT leak into the section text. No text is lost: the
chapter preamble (before the first subchapter) comes out as a separate leaf `N.0` so it reaches Chunk.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from ingest.contracts import Section
from ingest.orchestrator import StepContext

# Structure profiles — domain knowledge about book formats (§4.2.1, steps 0→1). A profile = detectors:
#   `chapter`     — running-header regex (gives the chapter number+title; the title source);
#   `subsections` — subsection detection mode in the chapter body:
#       "footers"  — DDIA: split bottom footers `<name>`/`|`/`<page>` → section names (with page_range);
#       "numbered" — Manning: clean numbered body headings `N.M[.K]  Title` → cut points;
#   `opener`      — (optional) regex of the chapter's caps opening-page marker `CHAPTER N`. This is the TRUE
#       chapter boundary: in pdfminer it comes BEFORE the first running-header repeat, so it catches the
#       whole opener (the intro → `N.0` and the FIRST subheading → `N.1`), which otherwise leaked into the
#       previous chapter, while the first subchapter sank into the preamble (§4.2.1 hardening). No key → cut by header.
# `StructureStep` stays a mechanism: detectors are injected, the format names live here.
#   ddia    — pdfminer Designing_Data.pdf: `Chapter N: Title` + footers + caps opener `CHAPTER N`;
#   manning — pdfminer Acing_System_Design_Interview: `chapter N  Title` (lowercase, two+ spaces,
#             no colon) + numbered (no caps opener in this layout);
#   progit  — pdfminer progit_en.pdf (Asciidoctor PDF): NEITHER a `Chapter N` header NOR
#             numbered body headings. The only reliable chapter signal is a line
#             starting with form-feed `\x0c` (new page) whose text EXACTLY equals a chapter
#             title from the curated set (the `chapter` regex matches on the RAW line, form-feed at the start).
#             The form-feed check resolves the collision: a chapter title appears in prose as a reference and
#             as a section heading in another chapter, but those lack the form-feed. Pro Git section headings are
#             bare title-case lines without markers → not reliably detectable; mode `none` = chapter-level
#             structure (chapter = leaf, chunked further by the chunker; TOC-driven subsections — §4.2.1 step 2).
BOOK_PROFILES: dict[str, dict[str, str]] = {
    "ddia": {
        "chapter": r"^Chapter\s+(\d+):\s+(.+)$",
        "subsections": "footers",
        "opener": r"^CHAPTER\s+(\d+)$",
    },
    "manning": {"chapter": r"^chapter\s+(\d+)\s{2,}(.+)$", "subsections": "numbered"},
    "progit": {
        "chapter": (
            r"^\f[ \t]*("
            r"Getting Started|Git Basics|Git Branching|Git on the Server|Distributed Git|"
            r"GitHub|Git Tools|Customizing Git|Git and Other Systems|Git Internals"
            r")[ \t]*$"
        ),
        "subsections": "none",
    },
}
DEFAULT_PROFILE = "ddia"
_DEFAULT_CHAPTER_RE = BOOK_PROFILES[DEFAULT_PROFILE]["chapter"]
_DEFAULT_SUBSECTION_MODE = BOOK_PROFILES[DEFAULT_PROFILE]["subsections"]
_DEFAULT_OPENER = BOOK_PROFILES[DEFAULT_PROFILE].get("opener")
# Book-level back-matter: headings a legitimate chapter subchapter is NEVER called.
# Their footers attach to the last chapter (cur_chapter doesn't change) → noise in the tree and big-block
# (Index = a wall of "term, page"). The body of the LAST chapter is cut at the first such heading.
DEFAULT_BACK_MATTER: frozenset[str] = frozenset(
    {"Glossary", "Index", "Bibliography", "Colophon"}
)
_PIPE = "|"

# Numbered body heading (Manning): `N.M` or `N.M.K`, exactly two spaces, then a letter-led title.
# Two spaces is the key discriminator (pdfminer renders the number→title gap as 2 spaces),
# cutting off prose ("1.5 million"), versions and numbered lists ("10. item").
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)+ {2}([A-Za-z].*)$")

# progit (mode `none`): form-feed opener of back-matter — appendices/index. On meeting one, we stop
# collecting body (cur=None), otherwise Appendix A/B/C, Index, etc. leak into the last chapter.
_PROGIT_BACK_MATTER_RE = re.compile(
    r"^\f[ \t]*(?:Appendix\b|Index[ \t]*$|Glossary[ \t]*$|Bibliography[ \t]*$)"
)
# A lone page number (pdfminer renders these as separate numeric lines) — noise, doesn't go into the body.
_LONE_PAGENUM_RE = re.compile(r"^\s*\d{1,4}\s*$")


def resolve_profile(name: str) -> dict[str, str]:
    """Profile name → {chapter, subsections} from the registry. Unknown — an explicit error, not a "silent" default."""
    try:
        return BOOK_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown book profile {name!r}; known: {sorted(BOOK_PROFILES)}"
        ) from None


class StructureError(RuntimeError):
    """Input with no chapters (or empty) — the step fails rather than write an empty sections.jsonl."""


def _count_tokens(text: str) -> int:
    """Approximate token counter at the Structure stage (exact tiktoken — at Chunk, S10)."""
    return len(text.split())


class StructureStep:
    name: str = "02_structure"
    artifacts: Sequence[str] = ("02_structure/sections.jsonl",)

    def __init__(
        self,
        *,
        external_id: str,
        chapter_pattern: str = _DEFAULT_CHAPTER_RE,
        subsection_mode: str = _DEFAULT_SUBSECTION_MODE,
        opener_pattern: str | None = _DEFAULT_OPENER,
        back_matter: frozenset[str] | None = None,
    ) -> None:
        if subsection_mode not in ("footers", "numbered", "none"):
            raise ValueError(f"unknown subsection_mode {subsection_mode!r}")
        self._external_id = external_id
        self._chapter_pattern = chapter_pattern
        self._chapter_re = re.compile(chapter_pattern)
        self._subsection_mode = subsection_mode
        self._opener_pattern = opener_pattern
        self._opener_re = re.compile(opener_pattern) if opener_pattern else None
        self._back_matter = (
            DEFAULT_BACK_MATTER if back_matter is None else frozenset(back_matter)
        )

    def params(self) -> dict:
        # Restart slice: changed book/detector (chapter/subsections/opener/back-matter) → hash
        # mismatch → re-cut as a cascade (S6).
        return {
            "external_id": self._external_id,
            "chapter_pattern": self._chapter_pattern,
            "subsection_mode": self._subsection_mode,
            "opener_pattern": self._opener_pattern,
            "back_matter": sorted(self._back_matter),
        }

    # --- orchestrator step ------------------------------------------------------
    def run(self, ctx: StepContext) -> str:
        src = ctx.rundir.path("01_extract", "full_text.txt")
        text = src.read_text(encoding="utf-8")
        sections = self.parse(text)
        out = ctx.stage_dir / "sections.jsonl"
        out.write_text(
            "\n".join(s.model_dump_json() for s in sections) + "\n",
            encoding="utf-8",
        )
        return str(out)

    # --- pure parsing (unit-testable, no disk) ---------------------------
    def parse(self, text: str) -> list[Section]:
        if self._subsection_mode == "none":
            return self._parse_chapter_level(text)  # progit: chapter level by form-feed
        lines = text.splitlines()
        # footers — only for the DDIA mode; numbered doesn't use them (the format has none).
        if self._subsection_mode == "footers":
            footer_noise, footers = self._scan_footers(lines)
        else:
            footer_noise, footers = set(), {}
        chapters = self._split_chapters(lines, footer_noise)
        if not chapters:
            raise StructureError(
                "No chapters found (running header "
                f"{self._chapter_pattern!r}) — nothing to structure."
            )

        sections: list[Section] = []
        for ch_ordinal, ch in enumerate(chapters, start=1):
            if self._subsection_mode == "footers":
                sub_names = self._ordered_subnames(ch["body"], footers.get(ch["num"], {}))
            else:  # numbered: clean numbered body headings — the cut points themselves
                sub_names = self._numbered_subheadings(ch["body"])
            # back-matter: only the LAST chapter can be trailed by Glossary/Index/… (cur_chapter
            # didn't change) — cut the body at the first such heading and drop them from subchapters.
            if self._back_matter and ch_ordinal == len(chapters):
                ch, sub_names = self._strip_back_matter(ch, sub_names)
            children = self._cut_subchapters(ch["body"], sub_names)
            chapter_text = self._join(ch["body"])

            sections.append(
                Section(
                    external_id=self._external_id,
                    path=str(ch["num"]),
                    depth=1,
                    parent_path=None,
                    ordinal=ch_ordinal,
                    heading=ch["title"],
                    is_leaf=not children,
                    text=chapter_text,
                    token_count=_count_tokens(chapter_text),
                    page_range=_span(footers.get(ch["num"], {}).values()),
                )
            )
            for child in children:
                page = footers.get(ch["num"], {}).get(child["heading"])
                sections.append(
                    Section(
                        external_id=self._external_id,
                        path=f"{ch['num']}.{child['ordinal']}",
                        depth=2,
                        parent_path=str(ch["num"]),
                        ordinal=child["ordinal"],
                        heading=child["heading"],
                        is_leaf=True,
                        text=child["text"],
                        token_count=_count_tokens(child["text"]),
                        page_range=(page, page) if page is not None else None,
                    )
                )
        return sections

    # --- progit: chapter-level structure (subsections="none") --------------------
    def _parse_chapter_level(self, text: str) -> list[Section]:
        """Pro Git: chapters by form-feed openers, each a leaf (no depth-2). See the `progit` profile."""
        chapters = self._split_chapters_by_titles(text)
        if not chapters:
            raise StructureError(
                "Pro Git: no chapter form-feed opener found "
                f"({self._chapter_pattern!r}) — nothing to structure."
            )
        sections: list[Section] = []
        for ch_ordinal, ch in enumerate(chapters, start=1):
            chapter_text = self._join(ch["body"])
            sections.append(
                Section(
                    external_id=self._external_id,
                    path=str(ch["num"]),
                    depth=1,
                    parent_path=None,
                    ordinal=ch_ordinal,
                    heading=ch["title"],
                    is_leaf=True,
                    text=chapter_text,
                    token_count=_count_tokens(chapter_text),
                    page_range=None,
                )
            )
        return sections

    def _split_chapters_by_titles(self, text: str) -> list[dict]:
        """Pro Git chapters = form-feed lines with an exact title from the set (chapter regex on the RAW line).

        `str.splitlines()` splits on form-feed and EATS it → the signal is lost; so here the split is
        strictly on `\\n` so that `\\f` survives to matching. Plain-page form-feeds are cut from the body;
        the body of the LAST chapter stops at the first form-feed back-matter (Appendix/Index).
        """
        chapters: list[dict] = []
        cur: dict | None = None
        seen: set[str] = set()
        for raw in text.split("\n"):
            line = raw.rstrip("\r")
            m = self._chapter_re.match(line)
            if m:
                title = m.group(1).strip()
                if title not in seen:  # title repeat (a form-feed reference is unlikely) — ignore
                    seen.add(title)
                    cur = {"num": len(chapters) + 1, "title": title, "body": []}
                    chapters.append(cur)
                continue
            if _PROGIT_BACK_MATTER_RE.match(line):
                cur = None  # appendices/index — stop, they don't go into the chapter body
                continue
            if cur is None:
                continue  # front-matter before chapter 1 / back-matter after the last
            cleaned = line.replace("\f", "")  # an ordinary page's form-feed — not let into the text
            if _LONE_PAGENUM_RE.match(cleaned):
                continue  # a lone page number — noise
            cur["body"].append(cleaned)
        return chapters

    # --- internal ------------------------------------------------------------
    def _scan_footers(self, lines: list[str]) -> tuple[set[int], dict[int, dict[str, int]]]:
        """Reconstruct footers `<name>` / `|` / `<digits>` over the stream of NON-EMPTY lines.

        Returns: indices of noise lines (footer fragments) and {chapter_num: {name: page}}.
        The name is attributed to the chapter active at the footer's moment (by the running header).
        """
        ne = [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]
        noise: set[int] = set()
        footers: dict[int, dict[str, int]] = {}
        cur_chapter: int | None = None
        for k in range(len(ne)):
            idx, val = ne[k]
            m = self._chapter_re.match(val)
            if m:
                cur_chapter = int(m.group(1))
                continue
            if k + 2 < len(ne):
                (_, b), (i_c, c) = ne[k + 1], ne[k + 2]
                if b == _PIPE and c.isdigit() and self._looks_like_heading(val):
                    footers.setdefault(cur_chapter or 0, {})[val] = int(c)
                    noise.update({ne[k][0], ne[k + 1][0], i_c})
        return noise, footers

    def _running_titles(self, lines: list[str]) -> dict[int, str]:
        """Canonical chapter titles from the running header `{num: title}` (first appearance)."""
        titles: dict[int, str] = {}
        for raw in lines:
            m = self._chapter_re.match(raw.strip())
            if m:
                titles.setdefault(int(m.group(1)), m.group(2).strip())
        return titles

    def _split_chapters(self, lines: list[str], footer_noise: set[int]) -> list[dict]:
        """Split lines into chapters; the body is free of running-header/opener noise.

        If the text has caps openers `CHAPTER N` (and the profile defines `opener`) — the chapter boundary
        is on them (the true opener start, before the first header; §4.2.1 hardening). Otherwise — fall back to
        the running header `Chapter N:` (the old path; backward compatibility with the fixtures/manning).
        """
        use_opener = self._opener_re is not None and any(
            self._opener_re.match(ln.strip()) for ln in lines
        )
        if not use_opener:
            return self._split_by_running_header(lines, footer_noise)

        run_titles = self._running_titles(lines)  # take the title from the header (canonical)
        chapters: list[dict] = []
        cur: dict | None = None
        seen_nums: set[int] = set()
        skip_title_line = False  # don't let the opener's title line (next non-empty) into the body
        for i, raw in enumerate(lines):
            s = raw.strip()
            mo = self._opener_re.match(s)
            if mo:
                num = int(mo.group(1))
                skip_title_line = True
                if num not in seen_nums:
                    seen_nums.add(num)
                    cur = {"num": num, "title": run_titles.get(num, ""), "body": []}
                    chapters.append(cur)
                continue  # the opener caps marker — noise
            if skip_title_line:
                if not s:
                    continue  # blanks between marker and title — skip, wait for the title
                skip_title_line = False
                if cur is not None and not cur["title"]:
                    cur["title"] = s  # no header was present — title from the opener line
                continue  # the opener's display title — noise, doesn't go into the body
            if self._chapter_re.match(s):
                continue  # a running header in the body — noise
            if cur is None:
                continue  # front-matter before the first chapter — skip
            if i in footer_noise:
                continue  # a footer fragment (name/|/number) — noise
            cur["body"].append(raw)
        return chapters

    def _split_by_running_header(
        self, lines: list[str], footer_noise: set[int]
    ) -> list[dict]:
        """Fallback: chapter boundaries by the running header `Chapter N:` (the original detector §4.2)."""
        chapters: list[dict] = []
        cur: dict | None = None
        seen_nums: set[int] = set()
        for i, raw in enumerate(lines):
            m = self._chapter_re.match(raw.strip())
            if m:
                num, title = int(m.group(1)), m.group(2).strip()
                if num not in seen_nums:
                    seen_nums.add(num)
                    cur = {"num": num, "title": title, "body": []}
                    chapters.append(cur)
                continue  # header line — noise, doesn't go into the body
            if cur is None:
                continue  # front-matter before the first chapter — skip
            if i in footer_noise:
                continue  # a footer fragment (name/|/number) — noise
            cur["body"].append(raw)
        return chapters

    def _strip_back_matter(self, ch: dict, sub_names: list[str]) -> tuple[dict, list[str]]:
        """Cut the chapter body at the first back-matter heading; remove back-matter from subchapters.

        Scans body lines for an exact match with a back-matter name (Glossary/Index/…), cuts
        at the earliest — drops both the heading and everything after (Index is a wall of "term, page").
        Returns a copy of `ch` with a truncated body and a filtered `sub_names`.
        """
        body_stripped = [ln.strip() for ln in ch["body"]]
        positions = [k for k, s in enumerate(body_stripped) if s in self._back_matter]
        if not positions:
            return ch, sub_names
        cut = min(positions)
        trimmed = {**ch, "body": ch["body"][:cut]}
        kept = [n for n in sub_names if n not in self._back_matter]
        return trimmed, kept

    def _numbered_subheadings(self, body: list[str]) -> list[str]:
        """Manning subsections: clean numbered body headings `N.M[.K]  Title` in order of appearance.

        Returns stripped heading lines (which are also the cut points for `_cut_subchapters`).
        Dedup in case of a repeated line. Dirty `N.M` where pdfminer split the number from the title
        don't get in here (the heading fragment merges into adjacent text — a known flaw §4.2.1).
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for ln in body:
            s = ln.strip()
            if _NUMBERED_HEADING_RE.match(s) and s not in seen:
                seen.add(s)
                ordered.append(s)
        return ordered

    def _ordered_subnames(self, body: list[str], footer_names: dict[str, int]) -> list[str]:
        """Subchapter names from footers, ordered by the FIRST appearance of the heading in the body."""
        body_stripped = [ln.strip() for ln in body]
        ordered = []
        for name in footer_names:
            if name in body_stripped:  # §4.2: the name appears in the body as a clean heading
                ordered.append((body_stripped.index(name), name))
        return [name for _, name in sorted(ordered)]

    def _cut_subchapters(self, body: list[str], sub_names: list[str]) -> list[dict]:
        """Cut the chapter body by subchapter heading lines; returns leaves in order."""
        body_stripped = [ln.strip() for ln in body]
        cuts = [(body_stripped.index(name), name) for name in sub_names]
        cuts.sort()
        children: list[dict] = []
        for ordinal, (start, name) in enumerate(cuts, start=1):
            end = cuts[ordinal][0] if ordinal < len(cuts) else len(body)
            text = self._join(body[start + 1 : end])  # without the heading line itself
            children.append({"ordinal": ordinal, "heading": name, "text": text})
        # preamble (text before the first subchapter) — a separate leaf N.0, so as not to lose text
        if cuts:
            preamble = self._join(body[: cuts[0][0]])
            if preamble:
                children.insert(0, {"ordinal": 0, "heading": "(preamble)", "text": preamble})
        return children

    @staticmethod
    def _looks_like_heading(s: str) -> bool:
        """Rough filter for a section name in a footer: a heading, not prose/number/boilerplate."""
        if not s or s[0].isdigit() or s == _PIPE:
            return False
        return s[0].isupper() and 1 <= len(s.split()) <= 10 and not s.endswith((".", ",", ";"))

    @staticmethod
    def _join(lines: list[str]) -> str:
        return "\n".join(ln.rstrip() for ln in lines).strip()


def _span(pages) -> tuple[int, int] | None:
    vals = [p for p in pages if isinstance(p, int)]
    return (min(vals), max(vals)) if vals else None
