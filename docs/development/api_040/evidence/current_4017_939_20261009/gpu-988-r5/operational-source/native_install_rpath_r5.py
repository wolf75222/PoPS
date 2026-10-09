"""Read-only GH200 ELF install proof; accepts only bounded CMake RPATH edits."""
import hashlib
import json
import re
import struct
import zipfile
from pathlib import Path


class InstallProofError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InstallProofError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def elf(data):
    require(len(data) >= 64 and data[:7] == b'\x7fELF\x02\x01\x01', 'ELF64 little endian required')
    h = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    require(h[1:4] == (3, 183, 1) and h[8] == 64 and h[9] == 56, 'AArch64 ET_DYN required')
    phoff, shoff, phnum, shnum, shstr = h[5], h[6], h[10], h[12], h[13]
    require(h[11] == 64 and 0 < phnum < 4096 and 0 < shnum < 16384 and 0 < shstr < shnum,
            'ordinary complete ELF tables required')
    def span(offset, size):
        require(0 <= offset <= len(data) and 0 <= size <= len(data)-offset, 'ELF range outside file')
        return data[offset:offset+size]
    span(phoff, phnum*56); span(shoff, shnum*64)
    ph = [struct.unpack_from('<IIQQQQQQ', data, phoff+i*56) for i in range(phnum)]
    sh = [struct.unpack_from('<IIQQQQIIQQ', data, shoff+i*64) for i in range(shnum)]
    names = span(sh[shstr][4], sh[shstr][5])
    def string(table, offset):
        require(0 <= offset < len(table), 'ELF string offset outside table')
        end = table.find(b'\0', offset)
        require(end >= 0, 'unterminated ELF string')
        return table[offset:end]
    sections = {}
    occupied = [(0,64),(phoff,phoff+phnum*56),(shoff,shoff+shnum*64)]
    for i, row in enumerate(sh):
        name = string(names, row[0]).decode('ascii')
        if i and row[1] != 8 and row[5]:
            span(row[4],row[5]); occupied.append((row[4],row[4]+row[5]))
        if name:
            require(name not in sections, 'duplicate ELF section name')
            sections[name] = row
    ordered = sorted(occupied)
    require(all(a[1] <= b[0] for a,b in zip(ordered,ordered[1:])), 'overlapping ELF section/table ranges')
    require('.dynamic' in sections and '.dynstr' in sections, 'ELF dynamic sections required')
    dynamic, dynstr = sections['.dynamic'], sections['.dynstr']
    require(dynamic[1] == 6 and not dynamic[2] & 4 and dynamic[9] == 16 and dynamic[5] % 16 == 0 and
            dynstr[1] == 3 and not dynstr[2] & 4 and sh[dynamic[6]] == dynstr,
            'canonical nonexecutable dynamic string table required')
    pd = [p for p in ph if p[0] == 2]
    require(len(pd) == 1 and pd[0][2] == dynamic[4] and pd[0][5] == dynamic[5], 'PT_DYNAMIC/section mismatch')
    loads = [p for p in ph if p[0] == 1]
    for section in [dynamic,dynstr]:
        require(any(p[2] <= section[4] and section[4]+section[5] <= p[2]+p[5] and
                    section[3] == p[3]+section[4]-p[2] and not p[1] & 1 for p in loads),
                'dynamic section lacks nonexecutable load mapping')
    table = span(dynstr[4],dynstr[5])
    entries = []; ended = False
    for offset in range(dynamic[4],dynamic[4]+dynamic[5],16):
        tag, value = struct.unpack_from('<QQ',data,offset)
        if ended or tag == 0:
            require(tag == value == 0, 'nonzero ELF dynamic padding')
            ended = True
        else:
            require(tag in set(range(1,39)) | {0x6ffffef5,0x6ffffff0,0x6ffffff9,0x6ffffffa,
                    0x6ffffffb,0x6ffffffc,0x6ffffffd,0x6ffffffe,0x6fffffff,
                    0x6ffffefa,0x6ffffefb,0x6ffffefc,0x7ffffffd,0x7fffffff}, 'unsupported ELF dynamic tag')
            entries.append((tag,value))
    require(ended, 'missing DT_NULL')
    require([v for t,v in entries if t == 5] == [dynstr[3]] and
            [v for t,v in entries if t == 10] == [dynstr[5]], 'DT_STRTAB/STRSZ mismatch')
    rpaths = [(t,v) for t,v in entries if t in (15,29)]
    require(len(rpaths) <= 1, 'ambiguous RPATH/RUNPATH')
    # Every known dynamic-string consumer must remain outside the editable slot.
    refs = [v for t,v in entries if t in (1,14,0x6ffffefa,0x6ffffefb,0x6ffffefc,0x7ffffffd,0x7fffffff)]
    for row in sh:
        if row[6] != sh.index(dynstr) or row == dynamic:
            continue
        payload = span(row[4],row[5])
        if row[1] in (2,11):
            require(row[9] == 24 and len(payload)%24 == 0, 'malformed ELF symbols')
            refs.extend(struct.unpack_from('<I',payload,i)[0] for i in range(0,len(payload),24))
        elif row[1] in (0x6ffffffe,0x6ffffffd):
            require(0 < row[7] <= len(payload)//8, 'invalid ELF version count')
            offset = 0; seen = set()
            for _ in range(row[7]):
                require(offset not in seen, 'cyclic ELF version records'); seen.add(offset)
                if row[1] == 0x6ffffffe:
                    require(offset+16 <= len(payload),'malformed verneed')
                    version,count,file,aux,next_ = struct.unpack_from('<HHIII',payload,offset)
                    refs.append(file); aux_at = offset+aux; step = 16
                else:
                    require(offset+20 <= len(payload),'malformed verdef')
                    version,flags,idx,count,hash_,aux,next_ = struct.unpack_from('<HHHHIII',payload,offset)
                    aux_at = offset+aux; step = 8
                require(version == 1 and count > 0, 'unsupported ELF version record')
                for j in range(count):
                    require(aux_at+step <= len(payload),'malformed version auxiliary')
                    name, advance = struct.unpack_from('<II',payload,aux_at+(8 if step == 16 else 0))
                    refs.append(name)
                    require((j == count-1) == (advance == 0), 'version auxiliary chain mismatch')
                    aux_at += advance
                offset += next_
            require(next_ == 0,'version count mismatch')
        else:
            raise InstallProofError('unsupported dynstr-linked section')
    references = [(v,v+len(string(table,v))+1) for v in refs]
    path = string(table,rpaths[0][1]) if rpaths else b''
    return {'sections':sections,'dynamic':dynamic,'dynstr':dynstr,'entries':entries,
            'rpath':rpaths[0] if rpaths else None,'path':path,'references':references}


def paths(raw, allowed, padded=False):
    text = raw.decode('ascii')
    if padded: text = text.rstrip(':')
    parts = text.split(':') if text else []
    require(all(p and p.startswith('/') and str(Path(p).resolve()) in allowed for p in parts),
            'unauthorized or empty loader path')
    return parts


def compare_elf_rpath(linked, installed, old_rpath, new_rpath, allowed):
    require(new_rpath == '', 'only default empty CMake install RPATH admitted')
    before, after = elf(linked), elf(installed)
    require(len(linked) == len(installed) and before['sections'] == after['sections'],
            'ELF relayout/strip refused')
    oldpaths = paths(before['path'],allowed,True)
    remove = paths(old_rpath.encode(),allowed,True)
    expected = list(oldpaths)
    if remove:
        hits = [i for i in range(len(oldpaths)-len(remove)+1) if oldpaths[i:i+len(remove)] == remove]
        require(len(hits) == 1,'CMake OLD_RPATH is not one exact path sequence')
        del expected[hits[0]:hits[0]+len(remove)]
    require(paths(after['path'],allowed) == expected, 'installed RPATH differs from CMake contract')
    require(before['rpath'] is not None or after['rpath'] is None,'new RPATH introduced')
    if after['rpath']:
        require(after['rpath'] == before['rpath'], 'RPATH type or string offset changed')
    protected_before = [e for e in before['entries'] if e[0] not in (15,29)]
    protected_after = [e for e in after['entries'] if e[0] not in (15,29)]
    require(protected_before == protected_after,'dependency/dynamic metadata changed')
    masks = []; a = bytearray(linked); b = bytearray(installed)
    d = before['dynamic']; canonical = b''.join(struct.pack('<QQ',*e) for e in protected_before)
    canonical += bytes(d[5]-len(canonical))
    a[d[4]:d[4]+d[5]] = canonical; b[d[4]:d[4]+d[5]] = canonical
    if before['rpath']:
        begin = before['rpath'][1]; end = begin+len(before['path'])+1
        require(all(hi <= begin or lo >= end for lo,hi in before['references']+after['references']),
                'RPATH string aliases dependency/symbol/version names')
        start = before['dynstr'][4]+begin; stop = before['dynstr'][4]+end
        replacement = after['path']+b'\0' if after['rpath'] else b''
        require(len(replacement) <= stop-start, 'installed RPATH exceeds original slot')
        require(installed[start:stop] == replacement+bytes(stop-start-len(replacement)),
                'RPATH slot has nonzero leftover/payload bytes')
        a[start:stop] = bytes(stop-start); b[start:stop] = bytes(stop-start)
        masks.append({'section':'.dynstr','offset':start,'size':stop-start})
    require(a == b,'non-RPATH ELF byte changed (code/constants/symbols/CUDA/metadata)')
    dt = before['dynstr']; strings = linked[dt[4]:dt[4]+dt[5]]
    needed = [strings[v:strings.find(b'\0',v)].decode('ascii') for t,v in protected_before if t == 1]
    return {'linked_sha256':digest(linked),'installed_sha256':digest(installed),'needed_names':needed,
            'canonical_sha256':digest(a),'linked_loader_path':before['path'].decode(),
            'installed_loader_path':after['path'].decode(),'dynamic_nonpath_entries':protected_before,
            'normalization':[{'section':'.dynamic','offset':d[4],'size':d[5],
                             'rule':'ordered nonpath entries exact; DT_NULL padding zero'}]+masks,
            'protected_sections':{n:digest(linked[r[4]:r[4]+r[5]]) for n,r in before['sections'].items()
                                  if n not in ('.dynamic','.dynstr') and r[1] != 8},
            'scope':'Whole file exact after bounded load-path normalization; no ELF stripping or relayout'}


def authenticate_native_install(linked, installed, wheel, build, root, admission):
    linked, installed, wheel, build, root = map(Path,(linked,installed,wheel,build,root))
    cmake = build/'python/cmake_install.cmake'
    text = cmake.read_text()
    route = 'pops/_native/dim2/'+installed.name
    commands = re.findall(r'file\(RPATH_CHANGE\s+FILE "([^"\n]+)"\s+OLD_RPATH "([^"\n]*)"\s+NEW_RPATH "([^"\n]*)"\s*\)',text)
    matching = [c for c in commands if c[0] == '$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/'+route]
    require(len(matching) == 1,'one exact generated _pops RPATH_CHANGE required')
    allowed = {str(p.resolve()) for p in [root/'envs/pops_final_cuda_dim2/lib',
               root/'kokkos-unified-install/lib',Path(admission['cuda_cudart']['resolved']).parent,
               Path('/project/r250127/rmdraux/sol61-gpu-bootstrap-20261004/compiler13/lib')]}
    installed_bytes = installed.read_bytes()
    proof = compare_elf_rpath(linked.read_bytes(),installed_bytes,*matching[0][1:],allowed)
    wheel_sha = file_digest(wheel)
    with zipfile.ZipFile(wheel) as archive:
        require(archive.namelist().count(route) == 1 and archive.namelist().count('pops/_native/variants.json') == 1,
                'unique wheel Native and inventory required')
        require(archive.getinfo(route).file_size == len(installed_bytes), 'wheel Native size mismatch')
        require(archive.read(route) == installed_bytes,'wheel Native differs from actual installed bytes')
        require(archive.getinfo('pops/_native/variants.json').file_size <= 1048576,
                'wheel Native inventory exceeds bound')
        manifest = archive.read('pops/_native/variants.json')
        require(manifest == (installed.parent.parent/'variants.json').read_bytes(),
                'wheel inventory differs from actual installed inventory')
        doc = json.loads(manifest)
        require(doc.get('schema_version') == 1,'wheel Native inventory schema mismatch')
        rows = [r for r in doc['variants'] if r['dimension'] == 2]
        require(len(rows) == 1 and rows[0]['path'] == 'dim2/'+installed.name and
                rows[0]['sha256'] == digest(installed_bytes),'wheel Native inventory mismatch')
    require(file_digest(wheel) == wheel_sha, 'wheel changed during proof')
    proof.update({'cmake_install_path':str(cmake),'cmake_install_sha256':digest(cmake.read_bytes()),
                  'cmake_old_rpath':matching[0][1],'cmake_new_rpath':matching[0][2],
                  'authorized_resolved_directories':sorted(allowed),'wheel_native_member':route,
                  'wheel_sha256':wheel_sha,'wheel_native_sha256':digest(installed_bytes),
                  'wheel_inventory_sha256':digest(manifest)})
    return proof


def authenticate_loader_dependencies(linked_stdout, installed_stdout, needed_names):
    """Actual ldd paths and bytes, including transitive mathematical libraries, must match."""
    def inventory(text):
        result = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or re.fullmatch(r'linux-vdso\.so\.\d+\s+\(0x[0-9a-f]+\)',stripped):
                continue
            m = re.fullmatch(r'(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9a-f]+\)',stripped)
            direct = re.fullmatch(r'(/\S+)\s+\(0x[0-9a-f]+\)',stripped)
            require(m or direct, 'unparsed/missing actual loader dependency')
            name, path = (m[1],m[2]) if m else (Path(direct[1]).name,direct[1])
            require(name not in result,'duplicate actual loader dependency')
            try:
                resolved = Path(path).resolve(strict=True)
                result[name] = {'resolved':str(resolved),'sha256':file_digest(resolved)}
            except OSError as error:
                raise InstallProofError('actual loader dependency missing/unreadable: '+path) from error
        require(set(needed_names) <= set(result),'direct DT_NEEDED missing from actual loader inventory')
        return result
    linked, installed = inventory(linked_stdout),inventory(installed_stdout)
    require(linked == installed,'actual dependency resolution/bytes changed during installation')
    return installed
