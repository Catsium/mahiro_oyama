# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repo root for domain language and glossary terms.
- `docs/adr/` for architecture decision records that touch the area about to be changed.
- `DECISIONS.md` for existing decision history and bot behavior notes, especially when working on trading logic, data sources, exits, sizing, or learning loops.

If any of these files don't exist, proceed silently. Don't flag their absence or suggest creating them upfront. Producer skills such as `/grill-with-docs` can create them lazily when terms or decisions actually get resolved.

## Layout

This is a single-context repo:

```text
/
|-- CONTEXT.md
|-- DECISIONS.md
|-- docs/
|   `-- adr/
`-- src/
```

## Use the glossary's vocabulary

When output names a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term as defined in `CONTEXT.md`.

If the concept needed isn't in the glossary yet, either reconsider whether the wording belongs to this project or note the gap for `/grill-with-docs`.

## Flag ADR conflicts

If output contradicts an existing ADR or `DECISIONS.md`, surface it explicitly rather than silently overriding it.
