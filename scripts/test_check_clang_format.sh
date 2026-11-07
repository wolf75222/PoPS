#!/usr/bin/env bash
# Replays the cases of scripts/check_clang_format.sh in a throwaway git repository that holds the
# script under test and the probe files written below, under a style of its own. No case reads
# another file of this repository, so none depends on the state of the tree, and none is skipped.
#
#   bash scripts/test_check_clang_format.sh [SCRIPT]
#
# SCRIPT defaults to the check_clang_format.sh next to this file. The format job of ci.yml runs the
# bench when the script, the bench or a style file changes. It needs clang-format and
# git-clang-format 19.1.7 on PATH (pipx install clang-format==19.1.7). The developer's own git
# settings are kept out. Exit status: the number of failed checks.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_script=$(realpath "${1:-$here/check_clang_format.sh}")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
repo=$tmp/repo
git -c init.defaultBranch=probe init -q "$repo"
cd "$repo"
git config user.name probe
git config user.email probe@example.invalid
script=scripts/check_clang_format.sh
failed=0

# case_ NAME EXPECTED_EXIT COMMAND...: runs COMMAND, compares its exit status, keeps its output.
case_() {
  local name=$1 expect=$2
  shift 2
  set +e
  "$@" > "$tmp/out" 2>&1
  local got=$?
  set -e
  if [ "$got" -eq "$expect" ]; then
    echo "ok    $name (exit $got)"
  else
    echo "FAIL  $name: exit $got, expected $expect"
    tail -5 "$tmp/out" | sed 's/^/      /'
    failed=$((failed + 1))
  fi
}
fail() { echo "FAIL  $1"; failed=$((failed + 1)); }

# The probe repository: the script, a style with aligned trailing comments, an ignore list, and
# four headers. include/probe holds its own .clang-format, which inherits the root style.
#   plain.hpp        conforms;
#   conforming.hpp   a block of three aligned trailing comments, conforms;
#   unformatted.hpp  the same block, plus one non-conforming line elsewhere;
#   include/pinned/snapshot.hpp  does not conform, and .clang-format-ignore excludes it.
probe_bad='int   probe_bad_spacing=1;'
probe_good='// A conforming comment line.'
block() {
  printf '%s\n' '#pragma once' '' 'namespace probe {' '' \
    'inline void wide_overload(char*, unsigned long, int = 0) {}  // serial: identity' \
    'inline void narrow(char*, int) {}  // serial: identity' \
    'inline void middle(char*, unsigned long) {}  // serial: identity' '' '}  // namespace probe'
}
mkdir -p scripts include/probe include/pinned
cp "$source_script" "$script"
printf '%s\n' 'BasedOnStyle: Google' 'ColumnLimit: 100' 'AlignTrailingComments: true' > .clang-format
printf 'include/pinned/*.hpp\n' > .clang-format-ignore
printf 'BasedOnStyle: InheritParentConfig\n' > include/probe/.clang-format
printf '%s\n' '#pragma once' '' 'namespace probe {' 'int plain();' '}  // namespace probe' \
  > include/probe/plain.hpp
block > include/probe/conforming.hpp
block > include/probe/unformatted.hpp
clang-format -i --style=file include/probe/plain.hpp include/probe/conforming.hpp \
  include/probe/unformatted.hpp
sed -i "1a $probe_bad" include/probe/unformatted.hpp
printf '#pragma once\n%s\n' "$probe_bad" > include/pinned/snapshot.hpp
git add -A
git commit -qm "probe base"
git checkout -q --detach
probe=$(git rev-parse HEAD)
reset_probe() { git reset -q --hard; git clean -qfdx; git checkout -q --detach "$probe"; }
for f in include/probe/plain.hpp include/probe/conforming.hpp; do
  clang-format --dry-run --Werror --style=file "$f" >/dev/null 2>&1 || fail "probe $f does not conform"
done
! clang-format --dry-run --Werror --style=file include/probe/unformatted.hpp >/dev/null 2>&1 \
  || fail "probe include/probe/unformatted.hpp conforms"
echo "probe repository $(git rev-parse --short HEAD), $(clang-format --version)"

# Formats the changed lines only, as CONTRIBUTING.md asks for a file that did not conform
# (git clang-format wants the change staged).
format_changed_lines() {
  git add -u
  git clang-format --binary "$(command -v clang-format)" --style file >/dev/null 2>&1 || true
}
# A cascade case means something only if the file no longer conforms as a whole.
expect_cascade() {
  if clang-format --dry-run --Werror --style=file "$1" >/dev/null 2>&1; then
    fail "the probe made no alignment cascade in $1"
  fi
}
# The patch left by a failing check must be a plain a/ b/ diff of the file that git apply takes.
expect_patch() {
  if [ ! -s format-diff.patch ] || ! grep -q "^diff --git a/$1 b/$1\$" format-diff.patch \
      || grep -q $'\033' format-diff.patch || ! git apply --check format-diff.patch 2>/dev/null; then
    fail "$2: format-diff.patch is not a plain a/ b/ diff of $1 that applies"
  fi
}
shorten_wide() {
  sed -i 's/wide_overload(char\*, unsigned long, int = 0)/wide_overload(char*, unsigned long, int)/' "$1"
}
add_longer_overload() {
  sed -i '/narrow(char\*, int)/a inline void much_longer_overload_name(unsigned long long*, unsigned long) {}  // serial: identity' "$1"
}

# 1. Nothing changed since the base: passes and leaves no format-diff.patch.
reset_probe
case_ "1. no change since HEAD" 0 bash "$script" --changed-since HEAD
[ ! -e format-diff.patch ] || fail "1. format-diff.patch left after a pass"

# 2. A conforming line added to a file that does not conform elsewhere: passes.
reset_probe
sed -i "/^namespace probe/a $probe_good" include/probe/unformatted.hpp
case_ "2. conforming line in a file that does not conform elsewhere" 0 \
  bash "$script" --changed-since HEAD

# 3. A non-conforming line in that file: fails, and the patch changes that line only.
reset_probe
sed -i "/^namespace probe/a int   probe_more=2;" include/probe/unformatted.hpp
case_ "3. non-conforming changed line" 1 bash "$script" --changed-since HEAD
changed=$(grep -c '^[-+][^-+]' format-diff.patch 2>/dev/null) || true
if [ "${changed:-0}" -ne 2 ] || ! grep -q '^-int   probe_more=2;' format-diff.patch; then
  fail "3. the patch changes more than the changed line"
fi

# 4. The same line in a file that .clang-format-ignore excludes: not checked.
reset_probe
printf '%s\n' "int   probe_more=2;" >> include/pinned/snapshot.hpp
case_ "4. non-conforming line in an ignored file" 0 bash "$script" --changed-since HEAD

# 5. A new non-conforming file: every line of it is new, so it fails.
reset_probe
printf '#pragma once\n%s\n' "$probe_bad" > include/probe/new.hpp
git add -N include/probe/new.hpp
case_ "5. new non-conforming file" 1 bash "$script" --changed-since HEAD

# 6. The layout of ci.yml: a merge commit whose first parent is the base.
for kind in good bad; do
  reset_probe
  if [ "$kind" = good ]; then line=$probe_good; expect=0; else line="int   probe_more=2;"; expect=1; fi
  sed -i "/^namespace probe/a $line" include/probe/unformatted.hpp
  git commit -qam "probe change ($kind)"
  change=$(git rev-parse HEAD)
  git checkout -q --detach "$probe"
  git merge -q --no-ff --no-edit "$change"
  case_ "6. merge commit, $kind line, --changed-since HEAD^1" "$expect" \
    bash "$script" --changed-since HEAD^1
done

# 7. Refusals: an unknown base (shallow checkout), another clang-format version.
reset_probe
case_ "7. unknown base revision" 2 \
  bash "$script" --changed-since 0123456789abcdef0123456789abcdef01234567
mkdir -p "$tmp/fake"
printf '#!/bin/sh\necho "clang-format version 18.1.8"\n' > "$tmp/fake/clang-format"
chmod +x "$tmp/fake/clang-format"
case_ "7. clang-format 18.1.8 refused" 2 env PATH="$tmp/fake:$PATH" bash "$script" --changed-since HEAD

# 8. The whole tree, as quality.yml runs it: 1 with a non-conforming file, and a patch; 0 once
# every file conforms, and no patch. The ignored snapshot never counts.
reset_probe
case_ "8. whole tree, one file does not conform" 1 bash "$script"
[ -s format-diff.patch ] || fail "8. no format-diff.patch after a failure"
grep -q '^Non conformes : 1 / 4$' "$tmp/out" || fail "8. summary is not 'Non conformes : 1 / 4'"
reset_probe
clang-format -i --style=file include/probe/unformatted.hpp
case_ "8. whole tree, every file conforms" 0 bash "$script"
[ ! -e format-diff.patch ] || fail "8. format-diff.patch left after a pass"

# 9. T1b of the review: a longer overload added to the block of aligned comments of a file that
# conformed at the base, then the changed lines formatted: the neighbours stay misaligned, and the
# whole file is checked: fails.
reset_probe
add_longer_overload include/probe/conforming.hpp
format_changed_lines
expect_cascade include/probe/conforming.hpp
case_ "9. T1b: longer overload added to an aligned block" 1 bash "$script" --changed-since HEAD
expect_patch include/probe/conforming.hpp "9"

# 10. T1 of the review: one overload of the block shortened, then the changed line formatted:
# fails, since the file conformed at the base.
reset_probe
shorten_wide include/probe/conforming.hpp
format_changed_lines
expect_cascade include/probe/conforming.hpp
case_ "10. T1: shortened overload in an aligned block" 1 bash "$script" --changed-since HEAD
expect_patch include/probe/conforming.hpp "10"

# 11. The same cascade in a file that did not conform at the base: its changed lines only, passes.
reset_probe
shorten_wide include/probe/unformatted.hpp
format_changed_lines
expect_cascade include/probe/unformatted.hpp
case_ "11. cascade in a file that did not conform at the base" 0 bash "$script" --changed-since HEAD

# 12. T2 of the review: a non-conforming changed line in a file that did not conform at the base
# fails whatever the developer's git settings (prefixes, colour, external diff driver).
reset_probe
sed -i "/^namespace probe/a int   probe_more=2;" include/probe/unformatted.hpp
case_ "12. changed line, default git settings" 1 bash "$script" --changed-since HEAD
case_ "12. changed line, diff.noprefix and color.ui=always" 1 env GIT_CONFIG_COUNT=2 \
  GIT_CONFIG_KEY_0=diff.noprefix GIT_CONFIG_VALUE_0=true \
  GIT_CONFIG_KEY_1=color.ui GIT_CONFIG_VALUE_1=always bash "$script" --changed-since HEAD
expect_patch include/probe/unformatted.hpp "12"
case_ "12. changed line, external diff driver" 1 env GIT_CONFIG_COUNT=1 \
  GIT_CONFIG_KEY_0=diff.external GIT_CONFIG_VALUE_0=true bash "$script" --changed-since HEAD

# 13. A style change, here in a directory's own .clang-format: a file that conformed at the base
# no longer does, though no C++ line changed.
reset_probe
printf 'SpacesBeforeTrailingComments: 4\n' >> include/probe/.clang-format
case_ "13. style change, file that conformed at the base" 1 bash "$script" --changed-since HEAD

# 14. The entry removed from .clang-format-ignore: the snapshot, which does not conform, is checked.
reset_probe
: > .clang-format-ignore
case_ "14. ignore entry removed, the snapshot is checked" 1 bash "$script" --changed-since HEAD

# 15. Started from a subdirectory: the paths are those of the root, so the line of case 3 is found.
reset_probe
sed -i "/^namespace probe/a int   probe_more=2;" include/probe/unformatted.hpp
case_ "15. non-conforming changed line, script started from include/" 1 \
  bash -c 'cd include && exec bash ../scripts/check_clang_format.sh --changed-since HEAD'

# 16. A branch behind its base: what only the base changed is not the branch's change. Fork point:
# a non-conforming line in conforming.hpp; the base formats that file; the branch adds a conforming
# line to plain.hpp. Checked against the base, the branch passes: the check starts from the merge
# base.
reset_probe
sed -i "/^namespace probe/a int   probe_more=2;" include/probe/conforming.hpp
git commit -qam "probe fork point"
fork_point=$(git rev-parse HEAD)
clang-format -i --style=file include/probe/conforming.hpp
git commit -qam "probe base ahead"
ahead=$(git rev-parse HEAD)
git checkout -q --detach "$fork_point"
sed -i "/^namespace probe/a $probe_good" include/probe/plain.hpp
case_ "16. branch behind its base, checked against the base" 0 \
  bash "$script" --changed-since "$ahead"

echo "failed checks: $failed"
exit "$failed"
