"""Test DSO faults at real FieldSolver@2 dispatch; adapter contract @4."""

from tests.python.integration.native_loader.test_external_field_solver_runtime import _solver_source


def fault_source(manifest):
    source = _solver_source(
        manifest,
        extra_includes="#include <cstdint>\n#include <stdexcept>\n#include <sys/mman.h>\n#include <fcntl.h>\n#include <unistd.h>\n#include <fstream>\n#include <cstdlib>\n#include <new>",
        solve_observer_statement="++control->callbacks;",
    )
    source = source.replace(
        "struct State {",
        "struct Control { PopsFieldSolverApiV2 table; int mode; int callbacks; int writes; };\nControl* control = nullptr;\nstruct State {",
    )
    source = source.replace(
        "if (mask[point] == 1) solution[index] = 7.0;",
        "if (mask[point] == 1) { solution[index] = 7.0; ++control->writes; }",
    )
    source = source.replace(
        "  report->status = POPS_SOLVE_SOLVED_V2;",
        '  if (control->mode == 1) throw std::runtime_error("test FieldSolver throw after actual writes");\n  report->status = POPS_SOLVE_SOLVED_V2;',
    )
    source = source.replace(
        "const PopsComponentInterfaceEntryV1 entry", "PopsComponentInterfaceEntryV1 entry"
    )
    init = r"""struct ControlOwner {
  int fd = -1;
  ControlOwner() {
    static_assert(offsetof(Control, table) == 0);
    static_assert(sizeof(int) == 4);
    static_assert(offsetof(PopsFieldSolverApiV2, header) == 0);
    static_assert(offsetof(PopsComponentTableHeaderV1, struct_size) == 0);
    static_assert(sizeof(table.header.struct_size) == sizeof(std::uint32_t));
    const char* path = std::getenv("POPS_TEST_FIELD_CONTROL");
    if (!path || !*path) throw std::runtime_error("test control path absent");
    fd = ::open(path, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) throw std::runtime_error("test control creation refused");
    if (::ftruncate(fd, sizeof(Control))) throw std::runtime_error("test control sizing failed");
    void* memory = ::mmap(nullptr, sizeof(Control), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (memory == MAP_FAILED) throw std::runtime_error("test control mapping failed");
    control = new(memory) Control{};
    control->table = table;
    entry.table = &control->table;
    std::ofstream out(std::string(path) + ".layout.json");
    out << "{\"schema\":\"sol61.test-field-shared-table@1\",\"pid\":" << ::getpid()
        << ",\"bytes\":" << sizeof(Control) << ",\"table_bytes\":" << sizeof(table)
        << ",\"header_bytes\":" << sizeof(table.header)
        << ",\"mode\":" << offsetof(Control, mode)
        << ",\"callbacks\":" << offsetof(Control, callbacks)
        << ",\"writes\":" << offsetof(Control, writes) << "}";
    out.close();
    if (!out) throw std::runtime_error("test control layout write failed");
  }
  ~ControlOwner() { if (control) ::munmap(control, sizeof(Control)); if (fd >= 0) ::close(fd); }
};
"""
    source = source.replace(
        'extern "C" const PopsComponentApiV1* pops_component_interface_v1() {',
        init
        + '\nextern "C" const PopsComponentApiV1* pops_component_interface_v1() {\n  static ControlOwner owner;',
    )
    source += '\nextern "C" const void* pops_test_actual_table_address() { return entry.table; }\n'
    return source


class SharedTableControl:
    """TEST-only typed mapping; no second image and no callback-pointer edits."""

    def __init__(self, path):
        import json
        import mmap
        import os
        from pathlib import Path

        self.path = Path(path)
        self.layout = json.loads(Path(str(path) + ".layout.json").read_text())
        expected = {
            "schema",
            "pid",
            "bytes",
            "table_bytes",
            "header_bytes",
            "mode",
            "callbacks",
            "writes",
        }
        if (
            set(self.layout) != expected
            or self.layout["schema"] != "sol61.test-field-shared-table@1"
        ):
            raise ValueError("invalid shared table layout")
        if (
            any(type(self.layout[k]) is not int for k in expected - {"schema"})
            or self.layout["pid"] != os.getpid()
        ):
            raise ValueError("foreign shared table process")
        size = self.layout["bytes"]
        if not (0 < self.layout["header_bytes"] < self.layout["table_bytes"] <= size):
            raise ValueError("invalid table sizes")
        offsets = [self.layout[k] for k in ("mode", "callbacks", "writes")]
        if len(set(offsets)) != 3 or any(
            not self.layout["table_bytes"] <= offset <= size - 4 for offset in offsets
        ):
            raise ValueError("invalid control offsets")
        fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            if os.fstat(fd).st_size != size:
                raise ValueError("invalid control length")
            self.memory = mmap.mmap(fd, size)
        finally:
            os.close(fd)

    def arm(self, mode):
        import struct

        if type(mode) is not int or mode not in (0, 1, 2):
            raise ValueError("invalid test fault mode")
        struct.pack_into("=i", self.memory, self.layout["mode"], mode)
        struct.pack_into(
            "=I",
            self.memory,
            0,
            self.layout["header_bytes"] if mode == 2 else self.layout["table_bytes"],
        )
        self.memory.flush()

    def counts(self):
        import struct

        return {
            key: struct.unpack_from("=i", self.memory, self.layout[key])[0]
            for key in ("callbacks", "writes")
        }


def json_evidence(value):
    """Lossless JSON framing for CBOR identity bytes in actual component metadata."""
    if type(value) is bytes:
        return {"encoding": "hex", "bytes": value.hex()}
    if type(value) in (dict,):
        return {key: json_evidence(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [json_evidence(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("unsupported evidence value")
