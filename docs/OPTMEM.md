# OptMem in this repo

This repo uses standalone OptMem, the same setup as grok-agy, study and council-go.
OptMem is a small permanent memory for AI agents. One plain fact per line.

## How it works here

- `OPTMEM-SEED.md` (repo root) is the seed: one fact per line, at most 280 bytes.
  Blank lines and `#` headings are skipped.
- The memory itself is **not** in this repo. It lives on each machine at
  `~/.local/share/grok-agy/optmem/repos/<key>/memory`, where `<key>` comes from the git `origin`
  (`https://github.com/fantastic25hippo/colab.git`).
- All machines share one store per repo through the private repo
  `fantastic25hippo/optmem-memory` (branch `main`). It syncs on `note` and `import`.
- Because the repo is public: never copy memory contents, keys or account emails into it.

## Use it (Grok or agy, from inside a clone)

```
grok-agy-memo wake            # read memory, run first in every session
grok-agy-memo note "..."      # one note per session or durable decision
grok-agy-memo import FILE     # bulk-load dated lines (bootstrap only)
```

The seed is imported once, only while the store is empty. Editing the file later does not
re-import it. Never nap from the agent and never edit the memory folder by hand.
`grok-agy-memo` prints one line and does nothing inside a `grok-agy` run
(`GROK_AGY=1`, `GROK_AGY_JOB_DIR` or `MEMORY_DIR` set) and warns if the repo has no `origin`.

## One-time machine setup

`grok-agy-setup --install` (from the grok-agy repo) installs the global rule files
`~/.grok/rules/optmem.md` and `~/.gemini/config/rules/optmem.md`, plus the Grok SessionEnd hook
`~/.grok/hooks/optmem-fill.json`. Details: the grok-agy repo's `docs/OPTMEM.md`.
