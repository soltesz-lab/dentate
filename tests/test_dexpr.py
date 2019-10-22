import importlib
import sympy, sympy.abc

def viewitems(obj, **kwargs):
    """
    Function for iterating over dictionary items with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewitems", None)
    if func is None:
        func = obj.items
    return func(**kwargs)

class DExpr(object):
    def __init__(self, parameter, expr, consts=None):
        self.sympy = importlib.import_module('sympy')
        self.sympy_parser = importlib.import_module('sympy.parsing.sympy_parser')
        self.sympy_abc = importlib.import_module('sympy.abc')
        self.parameter = parameter
        self.expr = self.sympy_parser.parse_expr(expr)
        self.consts = {} if consts is None else consts
        self.feval = None
        self.__init_feval__()
        
    def __getitem__(self, key):
        return self.consts[key]

    def __setitem__(self, key, value):
        self.consts[key] = value
        self.__init_feval__()
    
    def __init_feval__(self):
        fexpr = self.expr
        for k, v in viewitems(self.consts):
            sym = self.sympy.Symbol(k)
            fexpr = fexpr.subs(sym, v)
        self.feval = self.sympy.lambdify(self.sympy_abc.x, fexpr, "numpy")

    def __call__(self, x):
        return self.feval(x)

test_expr = DExpr(sympy.abc.x, "a * x + b", {"a": 2.0, "b": 5.0})

print(test_expr(0.5))
test_expr['a'] = 20.0
print(test_expr(0.5))


