# Agent behavior rules

Reusable, tool-agnostic rules for how an AI coding agent should work in any
project that depends on `torch_jaekwon`.

How to use: treat each rule below as a checklist — follow every rule that
applies.

Structure: `##` is a big category, `###` is a specific rule inside it. Numbers
(e.g. "1.1") are stable handles so you can say "check rule 1.1". Add new big
categories as `## 3.`, `## 4.`, ... as the need arises.

---

## 1. Explaining and citing code

### 1.1 Explain existing code intuitively, anchored to the code

When explaining the logic of existing code, always:

- **Explain intuitively first** — describe what the code does and *why* (the
  intent / mental model), not a line-by-line paraphrase of syntax.
- **Point to the exact code** — for each piece of logic, cite the specific file
  and lines so the user can see where it lives.
- **Show the code** — include the relevant snippet in a code-reference block
  (start/end line + filepath) so the explanation sits next to the implementation.

Do not explain logic in the abstract without showing the code it refers to.

### 1.2 Cite clickable full paths

When pointing the user at code, cite the **absolute path with line number**
(e.g.
`/home/jaekwoni/personal/projects/ntd/NeMo/nemo/collections/speechlm2/models/calm_interleaved.py:54`),
not a repo-relative path, so the reference is command-clickable from the user's
terminal.

## 2. Making changes

### 2.1 Show the plan and get confirmation before multi-file changes

Before creating or editing many different files (a new feature, a new
module/dir layout, anything touching several files at once), FIRST present the
proposed file/directory structure — which files will be created vs edited, and a
one-line purpose for each — and WAIT for the user's explicit confirmation before
writing any code.

- Exploration (reading/searching files, answering questions) needs no
  confirmation — only the actual file writes/edits do.
- For a trivial single-file change, just do it; this rule targets multi-file work.
- Surface the open design decisions alongside the structure (use the question
  tool) so they can be resolved before, not after, implementation.
