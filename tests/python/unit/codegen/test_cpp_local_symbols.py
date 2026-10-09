"""Declarations and real common-emitter references share an injective scoped table."""
from concurrent.futures import ThreadPoolExecutor
import pytest
from pops.codegen.cpp_symbols import symbol_table,variable_scope,variable_identifier,variable_bindings,CONTRACT
from pops.codegen.cpp_writer import _cpp_expand,_cse_emit
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen._compiler_lowering import require_compiler_lowering
from pops._ir.expr import Var
from tests.python.support.public_cpp_symbol_case import build,NAMES

def test_table_preserves_valid_names_and_separates_collisions_kinds_and_internal_names():
    symbols={('aux',n) for n in NAMES}|{('cons','same'),('aux','same'),('prim','same')}
    table=symbol_table(symbols)
    assert CONTRACT=='cpp-local-symbols@2'
    assert len(set(table.values()))==len(table)
    assert table['aux','a_b']=='a_b'
    assert table['aux','a b']!=table['aux','a_b']
    assert table['aux','κ']!=table['aux','λ']
    assert table['aux','providers']!='providers'
    assert table['aux',NAMES[-1]]==NAMES[-1] # authored name resembles the escape candidate
    assert table==symbol_table(reversed(sorted(symbols)))
    with variable_scope(symbols):
        for kind,name in symbols:
            assert _cpp_expand(Var(name,kind),{})==table[kind,name]
        lines,outputs=_cse_emit([Var('a b','aux')+Var('a_b','aux')],'pops::Real','')
        text=' '.join(lines+outputs)
        assert table['aux','a b'] in text and table['aux','a_b'] in text
        lines,outputs=_cse_emit([Var('same','cons')+Var('same','aux')],'pops::Real','',
            materialize_all=True,scalar_bindings={('cons','same'):'captured_cons'})
        assert 'captured_cons' in ' '.join(lines+outputs)
        assert table['aux','same'] in ' '.join(lines+outputs)
        from pops._ir.expr import Minimum,Const
        hidden=Minimum(Var('a b','aux'),Const(1))
        bindings=variable_bindings((hidden,))
        lines,outputs,observed=_cse_emit([hidden],'pops::Real','',materialize_all=True,
            scalar_bindings=bindings,return_names=True)
        assert len(observed)==2 # Aux leaf is observed before IEEE min can hide its NaN

def test_scope_resets_on_failure_and_separates_concurrent_emitters():
    original=variable_identifier('a b','aux')
    with pytest.raises(RuntimeError):
        with variable_scope((('aux','a b'),('aux','a_b'))):
            assert variable_identifier('a b','aux')!=original
            raise RuntimeError('intentional scope exit')
    assert variable_identifier('a b','aux')==original
    def render(collide):
        with variable_scope((('aux','a b'),('aux','a_b')) if collide else (('aux','a b'),)):
            return _cpp_expand(Var('a b','aux'),{})
    with ThreadPoolExecutor(max_workers=2) as pool:
        left,right=pool.map(render,(True,False))
    assert left!=right and right==original

@pytest.mark.parametrize('names',(NAMES,tuple(reversed(NAMES)),('difference gain',)))
def test_actual_public_program_model_cpp_keeps_component_authority(names):
    resolved=build(names);graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    model=graph.model_for_block(resolved.blocks[0].name)
    impl=getattr(model,'_m',model)
    symbols=[('cons',n) for n in impl.cons_names]+[('prim',n) for n in impl.prim_defs]+[('aux',n) for n in impl._provider_components]
    table=symbol_table(symbols)
    cpp=emit_cpp_program(resolved.time,model_graph=graph)
    assert cpp==emit_cpp_program(resolved.time,model_graph=graph)
    native=require_compiler_lowering(model).native_loader_source(name='Symbols',
        consumer_owner_qid=resolved.blocks[0].instance_owner_qid,
        declare_auxiliary_providers=resolved.blocks[0].declares_auxiliary_providers)
    for name in names:
        assert 'const pops::Real '+table['aux',name]+' =' in cpp
    assert {k.component for k in model._auxiliary_provider_pack}==set(names)
    assert all(k.owner_qid==str(require_compiler_lowering(model).source_module.owner_path.canonical())
               for k in model._auxiliary_provider_pack)
    assert native and all('"%s"'%name in native for name in names if name.isascii())


@pytest.mark.parametrize("primitive", (False, True))
def test_public_component_collision_and_recovery_use_the_real_printer(primitive):
    resolved=build(state_names=("a b","a_b"),recovery=True,primitive=primitive)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    block=resolved.blocks[0]
    source=require_compiler_lowering(graph.model_for_block(block.name)).native_loader_source(
        name="RecoverySymbols",consumer_owner_qid=block.instance_owner_qid,
        declare_auxiliary_providers=block.declares_auxiliary_providers)
    assert "const pops::Real a b" not in source
    assert "const pops::Real derived gain" not in source
    assert '"a b"' in source and '"a_b"' in source
    assert "recovery_admissible" in source


@pytest.mark.parametrize("names", (("a\"b","a_b"),("a\\b","a_b"),("a\nb","a_b"),("State","a_b"),("Axis","Prim"),("κ","🚀")))
def test_public_metadata_names_remain_exact_and_locals_compile_without_types_aliasing(names):
    from pops.codegen.cpp_strings import cpp_string_expression
    resolved=build(state_names=names,primitive=True)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks);block=resolved.blocks[0]
    source=require_compiler_lowering(graph.model_for_block(block.name)).native_loader_source(
        name="PublicText",consumer_owner_qid=block.instance_owner_qid,
        declare_auxiliary_providers=block.declares_auxiliary_providers)
    assert ', '.join(cpp_string_expression(n,ensure_ascii=False) for n in names) in source
    assert "const pops::Real State =" not in source
    assert "const pops::Real Axis =" not in source


def test_canonical_cpp_literal_preserves_json_legacy_and_utf8_controls():
    import json
    from pops.codegen.cpp_strings import cpp_string_literal,cpp_string_expression,CONTRACT
    assert CONTRACT=="cpp-public-text@2"
    for name in ("plain","a\"b","a\\b","a\nb","κ",r"literal\u0001"):
        assert cpp_string_literal(name)==json.dumps(name)
    assert cpp_string_literal("🚀")==r'"\U0001f680"'
    assert cpp_string_literal("\x01")==r'"\001"'
    assert cpp_string_expression("a\0b")==r'std::string{"a\000b", 3}'


def test_analytic_provider_identity_and_exact_parameters_keep_embedded_nul_and_astral():
    from pops.codegen.cpp_strings import cpp_string_view_expression
    assert cpp_string_view_expression("a\0b")==r'std::string_view{"a\000b", 3}'
    for name in ("nul\0tail","🚀"):
        resolved=build((name,));graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks);block=resolved.blocks[0]
        source=require_compiler_lowering(graph.model_for_block(block.name)).native_loader_source(
            name="LauncherText",consumer_owner_qid=block.instance_owner_qid,
            declare_auxiliary_providers=block.declares_auxiliary_providers)
        assert r"\ud83d" not in source.replace(r"\\ud83d","")
        if "\0" in name:
            assert "pops::PreparedProviderIdentity{std::string_view{" in source
            assert "std::string{" in source
