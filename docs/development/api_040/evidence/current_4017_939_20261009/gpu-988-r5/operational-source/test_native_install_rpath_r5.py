"""Actual synthetic ELF/ZIP bytes; parsing only, never executes AArch64 or CUDA."""
import json
import struct
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from native_install_rpath_r5 import (InstallProofError, authenticate_native_install, authenticate_loader_dependencies,
                                 compare_elf_rpath, digest, elf)


def fixture(raw='/admitted/lib::::::::', tag=29, needed_offset=None, symbol_offset=None):
    data = bytearray(2048)
    names = b'\0.text\0.nv_fatbin\0.dynstr\0.dynsym\0.dynamic\0.shstrtab\0.gnu.version_r\0'
    strings = b'\0libmath.so\0science_symbol\0'
    path_at = len(strings); strings += raw.encode()+b'\0'
    strings += bytes(160-len(strings)); data[320:480] = strings
    data[192:256] = b'CODE_CONSTANTS_'+bytes(49)
    data[256:320] = b'CUDA_FATBIN_PAYLOAD_'+bytes(44)
    struct.pack_into('<IBBHQQ',data,504,12 if symbol_offset is None else symbol_offset,18,0,1,192,8)
    entries = [(1,1 if needed_offset is None else needed_offset),(5,320),(10,160),(6,480),(11,24)]
    if tag: entries.insert(1,(tag,path_at))
    for i,e in enumerate(entries): struct.pack_into('<QQ',data,544+i*16,*e)
    data[736:736+len(names)] = names
    sections = [('',0,0,0,0,0,0,0,0,0),
                ('.text',1,6,192,192,64,0,0,16,0),
                ('.nv_fatbin',1,2,256,256,64,0,0,8,0),
                ('.dynstr',3,2,320,320,160,0,0,1,0),
                ('.dynsym',11,2,480,480,48,3,1,8,24),
                ('.dynamic',6,3,544,544,160,3,0,8,16),
                ('.shstrtab',3,0,0,736,len(names),0,0,1,0)]
    for i,row in enumerate(sections):
        name = names.index(row[0].encode()+b'\0') if row[0] else 0
        struct.pack_into('<IIQQQQIIQQ',data,1024+64*i,name,*row[1:])
    ident = b'\x7fELF\x02\x01\x01'+bytes(9)
    struct.pack_into('<16sHHIQQQIHHHHHH',data,0,ident,3,183,1,0,64,1024,0,64,56,2,64,7,6)
    struct.pack_into('<IIQQQQQQ',data,64,1,6,0,0,0,2048,2048,4096)
    struct.pack_into('<IIQQQQQQ',data,120,2,6,544,544,544,160,160,8)
    return bytes(data), path_at


def installed_from(data, raw='', keep_tag=False):
    value = elf(data); result = bytearray(data)
    d = value['dynamic']; entries = [e for e in value['entries'] if e[0] not in (15,29)]
    if keep_tag: entries.insert(1,value['rpath'])
    result[d[4]:d[4]+d[5]] = b''.join(struct.pack('<QQ',*e) for e in entries)+bytes(d[5]-16*len(entries))
    begin = value['dynstr'][4]+value['rpath'][1]; size = len(value['path'])+1
    replacement = raw.encode()+b'\0' if keep_tag else b''
    result[begin:begin+size] = replacement+bytes(size-len(replacement))
    return bytes(result)


class RpathProofTests(unittest.TestCase):
    def setUp(self):
        self.a,self.path_at = fixture()
        self.b = installed_from(self.a)

    def proof(self,a=None,b=None,old='/admitted/lib::::::::'):
        return compare_elf_rpath(a or self.a,b or self.b,old,'',{'/admitted/lib','/compiler/lib'})

    def test_runpath_remove_protects_whole_file(self):
        result = self.proof()
        self.assertNotEqual(result['linked_sha256'],result['installed_sha256'])
        self.assertEqual(result['installed_loader_path'],'')
        self.assertIn('.nv_fatbin',result['protected_sections'])

    def test_rpath_remove(self):
        a,_ = fixture(tag=15); self.proof(a,installed_from(a))

    def test_real_cmake_edits_synthetic_elf_without_execution(self):
        cmake = shutil.which('cmake')
        self.assertIsNotNone(cmake,'CMake is required for actual byte-edit discriminator')
        for tag,raw,old in [(29,'/admitted/lib::::::::','/admitted/lib::::::::'),
                            (15,'/admitted/lib::::::::','/admitted/lib::::::::'),
                            (29,'/compiler/lib:/admitted/lib::::','/admitted/lib::::')]:
            with self.subTest(tag=tag,raw=raw),tempfile.TemporaryDirectory() as tmp:
                a,_=fixture(raw,tag); p=Path(tmp)/'synthetic.so';p.write_bytes(a)
                script=Path(tmp)/'edit.cmake'
                script.write_text('file(RPATH_CHANGE FILE "'+str(p)+'" OLD_RPATH "'+old+'" NEW_RPATH "")\n')
                run=subprocess.run([cmake,'-P',str(script)],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr)
                self.proof(a,p.read_bytes(),old)

    def test_retained_authorized_compiler_path(self):
        a,_ = fixture('/compiler/lib:/admitted/lib::::')
        b = installed_from(a,'/compiler/lib',True)
        result = self.proof(a,b,'/admitted/lib::::')
        self.assertEqual(result['installed_loader_path'],'/compiler/lib')

    def test_exact_no_edit(self):
        a,_ = fixture('/admitted/lib')
        self.proof(a,a,old='')

    def test_nonpath_byte_changes_refused(self):
        for name,offset in [('code',194),('constant',230),('CUDA',270),('symbol_value',520),
                            ('build_metadata',1800),('ELF_flags',48),('needed_name',322)]:
            with self.subTest(name=name),self.assertRaises(InstallProofError):
                b=bytearray(self.b); b[offset] ^= 1; self.proof(b=bytes(b))

    def test_dynamic_dependency_value_change_refused(self):
        b=bytearray(self.b);struct.pack_into('<Q',b,552,12)
        with self.assertRaises(InstallProofError): self.proof(b=bytes(b))

    def test_cuda_section_relayout_refused(self):
        b=bytearray(self.b); struct.pack_into('<Q',b,1024+2*64+32,257)
        with self.assertRaises(InstallProofError): self.proof(b=bytes(b))

    def test_strip_resize_refused(self):
        with self.assertRaises(InstallProofError): self.proof(b=self.b[:-1])

    def test_unapproved_loader_path_refused(self):
        a,_=fixture('/unadmitted/lib')
        with self.assertRaises(InstallProofError): self.proof(a,installed_from(a),old='/unadmitted/lib')

    def test_unapproved_install_rewrite_refused(self):
        b=installed_from(self.a,'/admitted/lib',True)
        with self.assertRaises(InstallProofError): self.proof(b=b)
        with self.assertRaises(InstallProofError):
            compare_elf_rpath(self.a,self.b,'/admitted/lib::::::::','/admitted/lib',{'/admitted/lib'})

    def test_needed_symbol_and_version_alias_refused(self):
        for kind in ('needed','symbol','version'):
            with self.subTest(kind=kind):
                a,_=fixture(needed_offset=self.path_at if kind=='needed' else None,
                            symbol_offset=self.path_at if kind=='symbol' else None)
                if kind=='version':
                    v=bytearray(a); names=elf(a)['sections']['.shstrtab']; name_table=a[names[4]:names[4]+names[5]]
                    struct.pack_into('<IIQQQQIIQQ',v,1024+7*64,name_table.index(b'.gnu.version_r'),
                                     0x6ffffffe,2,832,832,32,3,1,8,0)
                    struct.pack_into('<H',v,60,8)
                    struct.pack_into('<HHIII',v,832,1,1,self.path_at,16,0)
                    struct.pack_into('<IHHII',v,848,0,0,0,12,0)
                    a=bytes(v)
                with self.assertRaises(InstallProofError): self.proof(a,installed_from(a))

    def test_nonzero_path_slot_tail_refused(self):
        b=bytearray(self.b);b[320+self.path_at+5]=1
        with self.assertRaises(InstallProofError): self.proof(b=bytes(b))

    def test_runpath_kind_conversion_refused(self):
        a,_=fixture('/compiler/lib:/admitted/lib::::')
        b=bytearray(installed_from(a,'/compiler/lib',True));struct.pack_into('<Q',b,560,15)
        with self.assertRaises(InstallProofError): self.proof(a,bytes(b),old='/admitted/lib::::')

    def test_malformed_dynamic_and_alias_layout_refused(self):
        for offset,value in [(544+6*16+8,1),(1024+3*64+32,544)]:
            with self.subTest(offset=offset),self.assertRaises(InstallProofError):
                b=bytearray(self.b);struct.pack_into('<Q',b,offset,value);self.proof(b=bytes(b))

    def test_real_wheel_and_cmake_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); build=root/'build'; (build/'python').mkdir(parents=True)
            lib=root/'envs/pops_final_cuda_dim2/lib'; lib.mkdir(parents=True)
            filename='_pops.cpython-312-aarch64-linux-gnu.so'
            installed=root/'site/pops/_native/dim2'/filename; installed.parent.mkdir(parents=True)
            linked=root/'linked.so'
            a,_=fixture(str(lib)+'::::'); b=installed_from(a); linked.write_bytes(a); installed.write_bytes(b)
            script=build/'python/cmake_install.cmake'
            script.write_text('file(RPATH_CHANGE\n FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/pops/_native/dim2/'+filename+
                              '"\n OLD_RPATH "'+str(lib)+'::::"\n NEW_RPATH "")\n')
            wheel=root/'pops-test.whl'
            def save(payload, row_sha=None, duplicate=False):
                with zipfile.ZipFile(wheel,'w') as z:
                    name='pops/_native/dim2/'+filename; z.writestr(name,payload)
                    if duplicate: z.writestr(name,payload)
                    z.writestr('pops/_native/variants.json',json.dumps({'schema_version':1,'variants':[
                        {'dimension':2,'path':'dim2/'+filename,'sha256':row_sha or digest(payload)}]}))
            save(b)
            with zipfile.ZipFile(wheel) as z:
                (installed.parent.parent/'variants.json').write_bytes(z.read('pops/_native/variants.json'))
            args=(linked,installed,wheel,build,root,{'cuda_cudart':{'resolved':'/cuda/lib/libcudart.so'}})
            self.assertEqual(authenticate_native_install(*args)['wheel_native_sha256'],digest(b))
            for kind in ('payload','inventory','duplicate','cmake'):
                with self.subTest(kind=kind):
                    save(a if kind=='payload' else b,'0'*64 if kind=='inventory' else None,kind=='duplicate')
                    if kind=='cmake': script.write_text(script.read_text().replace('dim2/','dim1/'))
                    with self.assertRaises(InstallProofError): authenticate_native_install(*args)

    def test_actual_dependency_file_hashes_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'libmath.so';p.write_bytes(b'real synthetic dependency bytes')
            q=Path(tmp)/'alternative.so';q.write_bytes(p.read_bytes())
            text='linux-vdso.so.1 (0x123)\nlibmath.so => '+str(p)+' (0xabc)\n'
            self.assertEqual(authenticate_loader_dependencies(text,text,['libmath.so'])['libmath.so']['sha256'],digest(p.read_bytes()))
            for altered in (text.replace(str(p),str(q)),text.replace('libmath.so','other.so'),
                            'libmath.so => not found\n'):
                with self.assertRaises(InstallProofError): authenticate_loader_dependencies(text,altered,['libmath.so'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
