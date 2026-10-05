"""Material properties written as formulas in the case file.

A property is a number or a formula in T, the temperature of the subsystem it belongs to:

    "740 * T"                                   # electron heat capacity of Pt, J/(m^3 K)
    "where(T <= 120.7, 9*T**3*(0.016*exp(-0.05*T) + exp(-0.14*T)), 1.3e6*T**-1.6)"

Allowed: numbers, T, + - * / ** (or ^), exp, log, sqrt, abs, min, max, and where(condition, a, b)
with the comparisons < <= > >=.  A formula is read with sympy, so a heat capacity C(T) can be
integrated exactly into the energy U(T) the solver conserves; the same formula is then turned
into NumPy code (for output) and into UFL (for the variational form).
"""
from __future__ import annotations

from functools import reduce

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

T = sp.Symbol("T", positive=True)
_NAMES = {
    "T": T, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs, "min": sp.Min, "max": sp.Max,
    "where": lambda condition, a, b: sp.Piecewise((a, condition), (b, True)),
}
_GLOBALS = {"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational, "Symbol": sp.Symbol,
            "Function": sp.Function, "__builtins__": {}}


class FormulaError(ValueError):
    """A formula that cannot be read or used."""


def parse(value) -> sp.Expr:
    """A number or a formula string, as a sympy expression in T."""
    if isinstance(value, bool):
        raise FormulaError(f"{value!r} is not a number or a formula")
    if isinstance(value, (int, float)):
        return sp.Float(value)
    if not isinstance(value, str):
        raise FormulaError(f"{value!r} is not a number or a formula")
    try:
        expr = parse_expr(value, local_dict=dict(_NAMES), global_dict=dict(_GLOBALS),
                          transformations=standard_transformations + (convert_xor,))
    except Exception as err:  # sympy reports syntax problems with many exception types
        raise FormulaError(f"cannot read formula {value!r}: {err}") from None
    unknown = sorted(str(s) for s in expr.free_symbols if s != T)
    unknown += sorted({str(f.func) for f in expr.atoms(sp.core.function.AppliedUndef)})
    if unknown:
        raise FormulaError(f"formula {value!r} uses {', '.join(unknown)}; only T and "
                           "exp, log, sqrt, abs, min, max, where are allowed")
    return expr


class Formula:
    """A function of temperature usable with NumPy arrays and with UFL expressions."""

    def __init__(self, expr: sp.Expr, text: str = ""):
        self.expr, self.text = expr, text or str(expr)
        self._numpy = sp.lambdify(T, expr, modules="numpy")

    def __call__(self, temperature):
        if isinstance(temperature, (int, float, np.ndarray, np.generic)):
            value = self._numpy(np.asarray(temperature, dtype=float))
            return np.broadcast_to(value, np.shape(temperature)).astype(float) if np.ndim(value) == 0 else value
        return to_ufl(self.expr, temperature)

    def integral(self) -> "Formula":
        """U(T) = integral of this function from 0 to T (a heat capacity -> an energy density)."""
        try:
            energy = sp.integrate(self.expr, (T, 0, T))
        except Exception:  # sympy raises several types when it cannot integrate
            energy = None
        if energy is None or energy.has(sp.Integral) or energy.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            raise FormulaError(f"cannot integrate the heat capacity {self.text!r} from 0 K to T; "
                               "write it so that it stays finite at 0 K")
        return Formula(sp.simplify(energy), f"integral of {self.text}")


def to_ufl(expr: sp.Expr, temperature):
    """The sympy expression `expr` with T replaced by the UFL expression `temperature`."""
    import ufl

    def conv(e):
        if e == T:
            return temperature
        if e.is_Number:
            return float(e)
        if isinstance(e, sp.Add):
            return reduce(lambda a, b: a + b, (conv(a) for a in e.args))
        if isinstance(e, sp.Mul):
            return reduce(lambda a, b: a * b, (conv(a) for a in e.args))
        if isinstance(e, sp.Pow):
            base, power = e.args
            return conv(base) ** (float(power) if power.is_Number else conv(power))
        if isinstance(e, sp.exp):
            return ufl.exp(conv(e.args[0]))
        if isinstance(e, sp.log):
            return ufl.ln(conv(e.args[0]))
        if isinstance(e, sp.Abs):
            return abs(conv(e.args[0]))
        if isinstance(e, sp.Min):
            return reduce(ufl.min_value, (conv(a) for a in e.args))
        if isinstance(e, sp.Max):
            return reduce(ufl.max_value, (conv(a) for a in e.args))
        if isinstance(e, sp.Piecewise):
            pieces = list(e.args)
            value = conv(pieces[-1][0])  # the last piece applies otherwise
            for piece, condition in reversed(pieces[:-1]):
                value = ufl.conditional(condition_to_ufl(condition), conv(piece), value)
            return value
        raise FormulaError(f"cannot use {e} in the solver")

    def condition_to_ufl(c):
        if c is sp.true:
            return ufl.ge(temperature, 0.0)
        ops = {sp.LessThan: ufl.le, sp.StrictLessThan: ufl.lt, sp.GreaterThan: ufl.ge, sp.StrictGreaterThan: ufl.gt}
        for kind, op in ops.items():
            if isinstance(c, kind):
                return op(conv(c.lhs), conv(c.rhs))
        if isinstance(c, sp.And):
            return reduce(ufl.And, (condition_to_ufl(a) for a in c.args))
        if isinstance(c, sp.Or):
            return reduce(ufl.Or, (condition_to_ufl(a) for a in c.args))
        raise FormulaError(f"cannot use the condition {c} in the solver")

    return conv(expr)
