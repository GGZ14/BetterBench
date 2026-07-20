# Contributing prompts

The corpus is what makes BetterBench *real-world*. Additions are welcome, under these rules so
results stay comparable and honest.

## Format
One JSON object per line in `corpus/v<version>/<category>.jsonl`:

```json
{"id":"code-gen-medium","category":"code","input_len_bucket":"short",
 "output_len_bucket":"medium","max_tokens":600,
 "messages":[{"role":"user","content":"..."}]}
```

- `id` — unique, `<category>-<shape>` style.
- `category` — matches the filename (e.g. `code.jsonl` → `"code"`).
- `input_len_bucket` / `output_len_bucket` — `short` | `medium` | `long`. Aim to fill the
  input×output grid per category (a file-edit over a big file is prefill-heavy; a story is
  decode-heavy).
- `max_tokens` — cap so runs are length-comparable within a cell.
- `messages` — OpenAI chat format; multi-turn allowed (see `chat.jsonl`).

## Rules
- **Representative of real work**, not adversarial or trick prompts.
- **License-clean & no PII.** Original or public-domain text only; no copyrighted passages, no
  real personal data.
- **Deterministic length target.** The prompt should reliably elicit roughly its output bucket so
  timing is comparable.
- **Speed-only.** v1 does not grade quality, so prompts don't need a reference answer — but they
  should be answerable so models actually generate (not refuse).

## Versioning
Changing existing prompts bumps `CORPUS_VERSION`. Results across corpus versions are **not**
comparable. Purely additive changes within a version are allowed only if they don't alter
existing prompts' ids or content.
