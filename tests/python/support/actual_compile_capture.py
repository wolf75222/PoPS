"""Test-only observation of actual compiler inputs, never source regeneration.

Contract actual-compile-capture@1 is local provenance, not Native qualification.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ActualCompileCapture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records = []

    def observe(self, original, command, purpose):
        if type(command) is not list or any(type(arg) is not str for arg in command):
            raise TypeError('compiler command must be an exact list of strings')
        sources = [Path(arg) for arg in command[1:] if arg.endswith('.cpp') and Path(arg).is_file()]
        if len(sources) != 1 or command.count('-o') != 1:
            raise ValueError('capture requires one actual C++ TU and one POSIX output')
        index = command.index('-o')
        if index + 1 == len(command):
            raise ValueError('compiler output is missing')
        output = Path(command[index + 1])
        compiler = shutil.which(command[0])
        if compiler is None:
            raise ValueError('compiler executable is unavailable')
        compiler = Path(compiler).resolve()
        signatures = [arg[len('-DPOPS_HEADER_SIG='):] for arg in command if arg.startswith('-DPOPS_HEADER_SIG=')]
        if len(signatures) != 1 or not signatures[0]:
            raise ValueError('actual SDK header signature must be unique and nonempty')
        source = sources[0].read_bytes()
        number = len(self.records)
        retained = self.directory / f'tu-{number}.cpp'
        retained.write_bytes(source)  # before invoking the unchanged compiler
        compiler_hash = sha(compiler)
        result = original(command, purpose)  # preserve the original exception/type
        if sources[0].read_bytes() != source or sha(compiler) != compiler_hash:
            raise ValueError('compiler or TU changed during compilation')
        if not output.is_file():
            raise ValueError('successful compiler did not produce its declared output')
        record = dict(contract='actual-compile-capture@1', purpose=purpose,
                      command=list(command), source_path=str(sources[0]),
                      retained_file=retained.name, source_sha256=sha(retained),
                      compiler_file=str(compiler), compiler_sha256=compiler_hash,
                      header_signature_argument=signatures[0], output_path=str(output),
                      output_sha256=sha(output), status='compiler-succeeded')
        (self.directory / f'tu-{number}.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        self.records.append(record)
        return result

    def require_binary(self, binary):
        """Authenticate a published binary, including atomic staging rename; no cache fallback."""
        digest = sha(binary)
        matches = [row for row in self.records if row['output_sha256'] == digest]
        if len(matches) != 1:
            raise ValueError('binary has no unique observed compiler TU (cache hit is not evidence)')
        row = matches[0]
        if sha(self.directory / row['retained_file']) != row['source_sha256']:
            raise ValueError('retained compiler TU changed')
        return dict(row)


@contextmanager
def capture_actual_compiles(directory, *, driver=None):
    if driver is None:
        from pops.codegen import _compile_drivers as driver
    capture = ActualCompileCapture(directory)
    original = driver._run_compile
    def observed(command, purpose):
        return capture.observe(original, command, purpose)
    driver._run_compile = observed
    try:
        yield capture
    finally:
        driver._run_compile = original
