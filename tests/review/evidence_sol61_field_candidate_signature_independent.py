"""Explicit external-evidence test; invoke with SOL61_EXPORT_PINS, never Native."""
import json
import os
from pathlib import Path
from tests.review.sol61_initial_field_ghost_saved_reader_v2 import receive


def test_actual_sdk14_export_is_received_without_signature_type_error():
    pins_path = Path(os.environ['SOL61_EXPORT_PINS'])
    authority = json.loads(pins_path.read_text())
    assert len(authority['pins']) == 28
    result = receive(Path(authority['case_directory']), authority['pins'])
    assert result['restart_full_carrier_and_Field_bits'] is True
    assert result['native_authority'] is False
    assert result['root_scientific_approval'] is False
