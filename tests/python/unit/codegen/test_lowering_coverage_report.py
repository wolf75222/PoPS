import pytest

from pops.codegen.lowering_coverage import LoweringCoverageReport, LoweringCoverageRow


def test_report_is_canonical_immutable_and_bidirectional():
    rows = (
        LoweringCoverageRow("z", "derived", ("target:b",), rule="z from b"),
        LoweringCoverageRow("a", "lowered", ("target:a", "target:b")),
        LoweringCoverageRow("m", "documentary"),
    )
    report = LoweringCoverageReport(rows)
    assert [row.source for row in report.rows] == ["a", "m", "z"]
    assert report.source_to_targets["a"] == ("target:a", "target:b")
    assert report.target_to_sources["target:b"] == ("a", "z")
    assert LoweringCoverageReport.from_data(report.to_data()).to_data() == report.to_data()
    with pytest.raises(TypeError):
        report.source_to_targets["new"] = ()
    with pytest.raises(AttributeError):
        report.rows = ()


@pytest.mark.parametrize("row, message", [
    (("a", "unknown", (), None, None), "disposition"),
    (("a", "lowered", (), None, None), "at least one target"),
    (("a", "derived", (), None, None), "derivation rule"),
    (("a", "documentary", ("target",), None, None), "cannot name behavior targets"),
    (("a", "rejected", ("target",), None, "gate"), "cannot name targets"),
    (("a", "rejected", (), None, None), "rejection gate"),
])
def test_row_validation_is_strict(row, message):
    with pytest.raises(ValueError, match=message):
        LoweringCoverageRow(*row)


def test_report_rejects_duplicate_sources_and_noncanonical_wire_order():
    with pytest.raises(ValueError, match="duplicate lowering source"):
        LoweringCoverageReport((
            LoweringCoverageRow("same", "documentary"),
            LoweringCoverageRow("same", "rejected", gate="unsupported"),
        ))
    data = LoweringCoverageReport((
        LoweringCoverageRow("a", "documentary"),
        LoweringCoverageRow("z", "documentary"),
    )).to_data()
    data["rows"].reverse()
    with pytest.raises(ValueError, match="canonical order"):
        LoweringCoverageReport.from_data(data)
