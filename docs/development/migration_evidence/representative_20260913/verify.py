"""Verify a copied curated evidence directory; does not execute PoPS or tests."""
from pathlib import Path
import argparse
import hashlib
import json

parser = argparse.ArgumentParser()
parser.add_argument('stage', type=Path)
parser.add_argument('--original-root', type=Path)
args = parser.parse_args()
manifest = json.loads((args.stage/'manifest.json').read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
expected = set()
for item in manifest['retained_files']:
    path = args.stage/item['path']
    assert path.is_file(), str(path)
    assert path.stat().st_size == item['bytes'], str(path)
    assert digest(path) == item['sha256'], str(path)
    expected.add(item['path'])
actual = {str(p.relative_to(args.stage)) for p in args.stage.rglob('*')
          if p.is_file() and p != args.stage/'manifest.json'}
assert actual == expected, {'unlisted':sorted(actual-expected),'missing':sorted(expected-actual)}
assert digest(args.stage/'index.json') == manifest['index_sha256']
if args.original_root:
    for item in manifest['files']:
        path = args.original_root/item['original_evidence_path']
        assert path.stat().st_size == item['original_bytes'], str(path)
        assert digest(path) == item['original_sha256'], str(path)
        if item.get('retained_path'):
            assert digest(args.stage/item['retained_path']) == item['retained_sha256']
index = json.loads((args.stage/'index.json').read_text())
for issue in index['issues']:
    for key in issue['representative_evidence_ids'] + issue['historical_evidence_ids']:
        assert key in index['evidence'], (issue['id'],key)
for entry in index['evidence'].values():
    for receipt in entry.get('receipts',[]):
        if receipt.get('receipt'):
            assert (args.stage/receipt['receipt']).is_file(), receipt
assert not any(p.suffix in {'.npz','.so','.dylib','.whl'} for p in args.stage.rglob('*'))
print(json.dumps({'verified_retained_files':len(expected),
                  'verified_original_files':len(manifest['files']) if args.original_root else None,
                  'issues':len(index['issues']), 'evidence_entries':len(index['evidence']),
                  'index_sha256':manifest['index_sha256']},indent=2))
