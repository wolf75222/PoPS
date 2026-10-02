"""Explicit pinned Native export audit; no native execution by these tests."""
import json,os
from pathlib import Path
from tests.review.sol61_initial_field_ghost_saved_reader_v3 import receive


def test_actual_typed_native_export_passes_scoped_reader():
    authority=json.loads(Path(os.environ['SOL61_TYPED_EXPORT_PINS']).read_text())
    assert len(authority['pins'])==28
    result=receive(authority['case_directory'],authority['pins'])
    assert result['restart_full_carrier_and_Field_bits'] is True
    assert result['native_authority'] is False and result['root_scientific_approval'] is False
