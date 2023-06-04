from SymMFEA.components.functions import FUNCTION_SET

class TestNode:
    def test_slots(*args):
        for fset in FUNCTION_SET.values():
            for f in fset:
                instance = f(arity = 2)
                assert not hasattr(instance, '__dict__'), str(instance)