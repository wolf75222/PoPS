#!/usr/bin/env bash
# Check C++ sources against .clang-format with the pinned clang-format 19.1.7.
#
#   bash scripts/check_clang_format.sh                         # every tracked C++ file
#   bash scripts/check_clang_format.sh --changed-since <rev>   # only the C++ lines changed since <rev>
#
# Run from the repository root. The format job of quality.yml (weekly on master) checks the whole
# tree; the format job of ci.yml checks, on every pull request and push, the lines that the change
# itself brings (`--changed-since HEAD^1`, the first parent of the merge commit that CI checks out),
# so it never asks a branch to reformat lines it does not touch. Both modes share the version, the
# extensions and .clang-format-ignore (byte-pinned snapshots, skipped by `clang-format --dry-run`
# and by git clang-format). The diff to apply goes to format-diff.patch (ignored by git, removed
# when the check passes) and the script exits 1.
# Fix it with `git clang-format --binary "$(command -v clang-format)" <base>`, or
# `clang-format -i <file>` for a whole file.
set -euo pipefail

version=$(clang-format --version)
echo "$version"
case "$version" in
  *"clang-format version 19.1.7"*) ;;
  *)
    echo "::error::clang-format 19.1.7 required (pipx install clang-format==19.1.7)" >&2
    exit 2
    ;;
esac

summary() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo "### clang-format ($1)"
      echo ""
      echo "- $2"
      echo "- Non conformes : **$3**"
      echo ""
      echo "_Corriger localement avec_ \`git clang-format --binary \"\$(command -v clang-format)\" <base>\`."
    } >> "$GITHUB_STEP_SUMMARY"
  fi
}

if [ "${1:-}" = "--changed-since" ]; then
  base=${2:?usage: check_clang_format.sh [--changed-since <rev>]}
  git rev-parse -q --verify "$base^{commit}" >/dev/null \
    || { echo "::error::clang-format : revision $base introuvable (fetch-depth ?)" >&2; exit 2; }
  command -v git-clang-format >/dev/null \
    || { echo "::error::git-clang-format required (installed with clang-format 19.1.7)" >&2; exit 2; }
  # git clang-format formats only the lines that differ from $base; --diff prints what it would
  # change and exits 1 when that is not empty.
  status=0
  git clang-format --binary "$(command -v clang-format)" --style file \
    --extensions hpp,h,cpp,cc,cxx,c --diff "$base" > format-diff.patch || status=$?
  if [ "$status" -gt 1 ]; then
    cat format-diff.patch
    echo "::error::git clang-format failed (status $status)" >&2
    exit 2
  fi
  bad=0
  if [ "$status" -eq 1 ]; then
    while IFS= read -r f; do
      echo "::warning file=$f::clang-format : lignes changees non conformes au style (.clang-format)"
      bad=$((bad+1))
    done < <(sed -n 's|^diff --git a/\(.*\) b/.*$|\1|p' format-diff.patch)
    cat format-diff.patch
  else
    rm -f format-diff.patch
  fi
  summary "lignes changees depuis $base" "Base : \`$(git rev-parse --short "$base")\`" "$bad"
  echo "Fichiers aux lignes changees non conformes : $bad"
  if [ "$bad" -ne 0 ]; then
    echo "::error::clang-format : $bad fichier(s) avec des lignes changees non conformes (.clang-format)"
    exit 1
  fi
  exit 0
fi

# Track all native source directories, including src/runtime and benchmarks.
# Git excludes generated build trees; NUL delimiters preserve paths with spaces.
mapfile -d '' -t files < <(git ls-files -z -- '*.hpp' '*.h' '*.cpp' '*.cc' '*.cxx' '*.c')
bad=0
: > format-diff.patch
for f in "${files[@]}"; do
  if clang-format --dry-run --Werror --style=file "$f" 2>/dev/null; then
    continue
  fi
  diff -u "$f" <(clang-format --style=file "$f") >> format-diff.patch || true
  echo "::warning file=$f::clang-format : non conforme au style (.clang-format)"
  bad=$((bad+1))
done
summary "arbre entier" "Fichiers analyses : ${#files[@]}" "$bad"
echo "Non conformes : $bad / ${#files[@]}"
if [ "$bad" -ne 0 ]; then
  echo "::error::clang-format : $bad fichier(s) non conforme(s) au style (.clang-format)"
  exit 1
fi
rm -f format-diff.patch
