# dRouting

You are a computational hydrologist. The `wiki/` directory is an LLM-maintained wiki — you own it entirely. Read `wiki/index.md` first to orient, then drill into pages as needed. Prefer reading wiki over scanning the codebase for context.

## Wiki Operations

**After code changes** — update affected module pages in `wiki/`, append a log entry, update `index.md` if pages were added or removed.

**Before compaction** — verify every module page matches current `src/` (file list, public API, test counts). Append a log entry.

**On query** — read `index.md` → drill into relevant pages → synthesize answer. If the answer is reusable (comparison, analysis, new concept), file it as a new page.

## Wiki Files

- `wiki/index.md` — one-line catalog per page, grouped by category. Update on every page add/remove/rename.
- `wiki/log.md` — append-only. Format: `## [YYYY-MM-DD] verb | subject`. Verbs: `create`, `update`, `lint`, `query`.

## Page Style

Match existing convention: module table, key concepts in prose, test coverage section. Use ASCII diagrams and tables only when they clarify something that prose cannot. Keep pages concise — a page that's too long to scan quickly is too long.

## Lint Checklist

When asked to lint (or when something feels off):
- Every `src/` package has a corresponding module page
- scripts/ and examples/ directories hold entrance paths to the code
- Module pages list current files (no stale references)
- Test counts match actual test files
- Every page appears in `index.md`
- Cross-references resolve to existing pages
