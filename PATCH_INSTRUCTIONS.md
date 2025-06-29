# PATCH_INSTRUCTIONS.md  
**How to apply `distributed-atomspace-integration.patch` to CogPrime**

---

## 1.  What the Patch Contains
`distributed-atomspace-integration.patch` (≈ 53 MB) bundles the **entire** node9 and mem0 code-bases plus glue code required for the unified distributed cognition stack:

| Directory added by the patch | Purpose |
| ---------------------------- | -------- |
| `integrations/node9/`        | Plan 9-style micro-kernel, Styx server, LuaJIT runtime |
| `integrations/mem0/`         | Vector & graph memory, persistence layer, LLM tools |
| `src/atomspace/`             | Full Python AtomSpace interface |
| `src/bridges/`               | Node9 ⇄ AtomSpace Styx bridge |
| `src/memory/`                | mem0 wrapper for CogPrime |
| plus updated tests, examples, docs, and build files |

Applying the patch to a **fresh clone** of `cogprime` recreates the exact integration used in PR #5 without requiring multi-repo cloning or submodules.

---

## 2.  Prerequisites
1. **Disk space:** ≥ 500 MB free. Patch expands to ≈ 300 MB of source.  
2. **Git ≥ 2.30** with `core.fileMode` enabled (default).  
3. A clean working tree (`git status` shows no changes).  
4. If you use Git LFS, ensure it’s up-to-date, but LFS is *not* required for the patch itself.

---

## 3.  Quick-Start (copy-paste)
```bash
# 1 – clone CogPrime and enter repo
git clone https://github.com/drzo/cogprime.git
cd cogprime

# 2 – create integration branch
git checkout -b feature/distributed-atomspace-integration

# 3 – download patch (place in repo root)
#    e.g. via browser download panel or:
# curl -L -o distributed-atomspace-integration.patch \
#      https://<YOUR_DOWNLOAD_URL>

# 4 – apply patch
git apply --stat distributed-atomspace-integration.patch   # optional preview
git am    --committer-date-is-author-date \
          --keep-non-patch distributed-atomspace-integration.patch
```
`git am` preserves author info and splits the patch into its original logical commits.  
If you prefer a single squashed commit, use `git apply` followed by `git add . && git commit`.

---

## 4.  Verifying the Patch
```bash
# Confirm directories were created
ls integrations/node9 | head
ls integrations/mem0 | head

# Run the simple smoke-test
python examples/simple_integration_test.py
```
You should see output ending with “**TEST COMPLETE**” and no errors.

---

## 5.  Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `error: patch failed` | Ensure repository state is clean and you are on the correct branch. |
| Large-file push freezes | Use `push_files_to_github_gitlab` helper (Factory) or split commits. |
| Line-ending conflicts on Windows | Add `--ignore-space-change --ignore-whitespace` to `git apply`. |
| Styx / node9 FFI errors | Build node9 per `integrations/node9/README.md` before running bridge. |

---

## 6.  House-Keeping Tips
* **Remove original git histories:** node9 & mem0 `.git` directories were already stripped in the patch; no further action needed.  
* **Optional squash:** After inspection, run `git reset --soft $(git merge-base main HEAD)` then commit once to reduce history size.  
* **Clean-up:** Once merged, delete the patch file and branch if desired.

---

## 7.  Need Help?
Open an issue or comment on Pull Request **#5** (`feature/distributed-atomspace-integration`)  
or ping `@drzo` in the CogPrime Discord.

