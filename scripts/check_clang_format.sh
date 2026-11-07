#!/usr/bin/env bash
# Check C++ sources against .clang-format with the pinned clang-format 19.1.7.
#
#   bash scripts/check_clang_format.sh                         # every tracked C++ file
#   bash scripts/check_clang_format.sh --changed-since <rev>   # what a change does to the C++ files
#
# The script works from the repository root, wherever it is started. The format job of quality.yml
# (weekly on master) checks the whole tree. The format job of ci.yml checks, on every pull request
# and push, what the change itself brings (`--changed-since HEAD^1`, the first parent of the merge
# commit that CI checks out), as a ratchet; when <rev> is not an ancestor of HEAD, the change starts
# from their merge base. A C++ file that the change adds or modifies passes when it conforms as a
# whole file. Otherwise it fails when it conformed at <rev>, since a changed line can leave its
# aligned neighbours behind (trailing comments, escaped newlines); a file that did not conform at
# <rev> is held to its changed lines only, so a branch never has to reformat a file that it found
# unformatted. When the change edits .clang-format or .clang-format-ignore, every tracked C++ file
# that conformed at <rev>, under the style and the ignore list of <rev>, must still conform. Both
# modes share the version, the extensions and .clang-format-ignore (byte-pinned snapshots, never
# checked). The diff to apply goes to format-diff.patch (ignored by git, removed when the check
# passes; `git apply` takes it) and the script exits 1.
# Fix a file with `clang-format -i <file>`, or, for a file that did not conform at <rev>,
# `git clang-format --binary "$(command -v clang-format)" <rev>` (its changed lines only).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

version=$(clang-format --version)
echo "$version"
case "$version" in
  *"clang-format version 19.1.7"*) ;;
  *)
    echo "::error::clang-format 19.1.7 required (pipx install clang-format==19.1.7)" >&2
    exit 2
    ;;
esac

cxx=('*.hpp' '*.h' '*.cpp' '*.cc' '*.cxx' '*.c')

summary() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo "### clang-format ($1)"
      echo ""
      echo "- $2"
      echo "- Non conformes : **$3**"
      echo ""
      echo "_Corriger localement avec_ \`clang-format -i <fichier>\`_, ou_ \`git clang-format\` _pour les lignes changees d'un fichier qui n'etait pas conforme._"
    } >> "$GITHUB_STEP_SUMMARY"
  fi
}

# Exits 0 when file $1 of the current checkout conforms as a whole file (.clang-format-ignore
# files conform: --dry-run skips them).
conforms() {
  clang-format --dry-run --Werror --style=file "$1" >/dev/null 2>&1
}

# Appends to format-diff.patch the diff that formats file $1 as a whole, in git's a/ b/ form.
file_diff() {
  {
    echo "diff --git a/$1 b/$1"
    diff -u --label "a/$1" --label "b/$1" "$1" <(clang-format --style=file "$1") || true
  } >> format-diff.patch
}

if [ "${1:-}" = "--changed-since" ]; then
  base=${2:?usage: check_clang_format.sh [--changed-since <rev>]}
  git rev-parse -q --verify "$base^{commit}" >/dev/null \
    || { echo "::error::clang-format : revision $base introuvable (fetch-depth ?)" >&2; exit 2; }
  # What the change itself brings: when <rev> is not an ancestor of HEAD (a branch behind its
  # base), the files that only the base changed would count as touched; start from the fork point.
  fork=$(git merge-base "$base" HEAD) \
    || { echo "::error::clang-format : pas de base de fusion entre $base et HEAD" >&2; exit 2; }
  if [ "$fork" != "$(git rev-parse "$base^{commit}")" ]; then
    echo "clang-format : $base n'est pas un ancetre de HEAD ; controle depuis la base de fusion $(git rev-parse --short "$fork")"
    base=$fork
  fi
  command -v git-clang-format >/dev/null \
    || { echo "::error::git-clang-format required (installed with clang-format 19.1.7)" >&2; exit 2; }
  scratch=$(mktemp -d)
  cleanup() {
    git worktree remove --force "$scratch/base" >/dev/null 2>&1 || true
    rm -rf "$scratch"
  }
  trap cleanup EXIT

  # The C++ files that the change adds or modifies (a renamed file counts as added).
  declare -A touched=()
  while IFS= read -r -d '' f; do
    touched[$f]=1
  done < <(git diff --no-ext-diff --no-relative --no-renames --name-only -z --diff-filter=AMT \
    "$base" -- "${cxx[@]}")
  # Exits 0 when file $1 conformed at $base. When the style and the ignore list are those of
  # $base, its blob is formatted here (a symbolic link never conforms); otherwise the file is
  # judged in a checkout of $base, with the style and the ignore list of $base.
  conformed_at_base() {
    if [ -n "$style_changed" ]; then
      (cd "$scratch/base" && [ -f "$1" ] && [ ! -L "$1" ] && conforms "$1")
    else
      case $(git ls-tree "$base" -- "$1") in
        100644\ *|100755\ *) ;;
        *) return 1 ;;
      esac
      git show "$base:$1" | clang-format --style=file --assume-filename="$1" \
        | cmp -s - <(git show "$base:$1")
    fi
  }
  style_changed=
  if git diff --no-ext-diff --quiet "$base" -- .clang-format _clang-format '*/.clang-format' \
      '*/_clang-format' .clang-format-ignore '*/.clang-format-ignore'; then
    candidates=("${!touched[@]}")
  else
    style_changed=1
    echo "clang-format : le style ou la liste d'exclusion change depuis $base ; chaque fichier conforme a la base doit le rester"
    git worktree add -q --detach "$scratch/base" "$base"
    mapfile -d '' -t candidates < <(git ls-files -z -- "${cxx[@]}")
  fi
  # Regular files only, in a stable order, without those that .clang-format-ignore excludes.
  declare -A ignored=()
  if [ "${#candidates[@]}" -gt 0 ]; then
    while IFS= read -r f; do
      ignored[$f]=1
    done < <(clang-format -list-ignored "${candidates[@]}")
  fi
  files=()
  while IFS= read -r -d '' f; do
    if [ -f "$f" ] && [ ! -L "$f" ] && [ -z "${ignored[$f]:-}" ]; then
      files+=("$f")
    fi
  done < <(for f in "${candidates[@]}"; do printf '%s\0' "$f"; done | sort -z)
  echo "Fichiers C++ controles : ${#files[@]}"

  : > format-diff.patch
  bad=0
  lines=()
  for f in "${files[@]}"; do
    if conforms "$f"; then
      continue
    fi
    if conformed_at_base "$f"; then
      file_diff "$f"
      echo "::warning file=$f::clang-format : conforme a la base, plus conforme en entier (lignes voisines a realigner ?) ; corriger avec clang-format -i $f"
      bad=$((bad+1))
    elif [ -n "${touched[$f]:-}" ]; then
      lines+=("$f")
    fi
  done
  # A file that did not conform at $base: git clang-format formats only the lines that differ
  # from $base, and --diff prints what it would change and exits 1 when that is not empty. One
  # call per file, so that its status alone gives the verdict; the settings that would change the
  # a/ b/ form or colour the diff are pinned.
  for f in "${lines[@]}"; do
    status=0
    git -c diff.noprefix=false -c diff.mnemonicPrefix=false -c diff.srcPrefix=a/ \
      -c diff.dstPrefix=b/ -c color.ui=never -c color.diff=never \
      clang-format --binary "$(command -v clang-format)" --style file \
      --extensions hpp,h,cpp,cc,cxx,c --diff "$base" -- "$f" > "$scratch/lines.patch" \
      || status=$?
    case $status in
      0) ;;
      1)
        cat "$scratch/lines.patch" >> format-diff.patch
        echo "::warning file=$f::clang-format : lignes changees non conformes au style (.clang-format)"
        bad=$((bad+1))
        ;;
      *)
        cat "$scratch/lines.patch"
        echo "::error::git clang-format failed on $f (status $status)" >&2
        exit 2
        ;;
    esac
  done
  summary "changement depuis $base" "Base : \`$(git rev-parse --short "$base")\`, ${#files[@]} fichier(s) controle(s)" "$bad"
  echo "Fichiers non conformes : $bad"
  if [ "$bad" -ne 0 ]; then
    cat format-diff.patch
    echo "::error::clang-format : $bad fichier(s) non conforme(s) au style (.clang-format)"
    exit 1
  fi
  rm -f format-diff.patch
  exit 0
fi

# Track all native source directories, including src/runtime and benchmarks.
# Git excludes generated build trees; NUL delimiters preserve paths with spaces.
mapfile -d '' -t files < <(git ls-files -z -- "${cxx[@]}")
bad=0
: > format-diff.patch
for f in "${files[@]}"; do
  if conforms "$f"; then
    continue
  fi
  file_diff "$f"
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
