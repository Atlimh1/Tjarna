"""Formulas the model writes and code evaluates, for calculations of more than one step:

    {"operation": "formula", "names": {"total": 0, "capital": 1}, "formula": "(total - capital) / total * 100"}
    {"operation": "formula", "names": {"old": [2, 3, 4], "all": 0}, "formula": "sum(old) / all * 100"}
    {"operation": "formula", "names": {"y2015": 0, "y2024": 9}, "formula": "cagr(y2015, y2024, 9)"}

The model only names rows and writes the formula; code checks that every name is a row it read (a list of rows
for sum/mean/...), parses the formula and evaluates + - * / ( ), numbers, the names and the functions in FUNCS. No
eval(), no attributes, no other names: anything else is refused and no calculation is shown.
"""
import ast
import math
import re
import statistics

MAX_FORMULA = 200  # characters
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,30}")


def _flat(args):
    out = []
    for a in args:
        out.extend(a if isinstance(a, list) else [a])
    return out


def _cagr(first, last, years):
    """Average yearly growth in %, from `first` to `last` over `years` years."""
    return ((last / first) ** (1 / years) - 1) * 100


FUNCS = {
    "sum": lambda *a: math.fsum(_flat(a)),
    "mean": lambda *a: statistics.fmean(_flat(a)),
    "median": lambda *a: statistics.median(_flat(a)),
    "min": lambda *a: min(_flat(a)),
    "max": lambda *a: max(_flat(a)),
    "abs": abs,
    "cagr": _cagr,
}
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
       ast.Div: lambda a, b: a / b}


class FormulaError(ValueError):
    pass


def evaluate(formula, names):
    """(result, intermediate results) of `formula` over `names` {name: float or [floats]}. FormulaError for
    anything outside the whitelist, an unknown name, a list where a number is needed, or a division by zero."""
    if not isinstance(formula, str) or not formula.strip() or len(formula) > MAX_FORMULA:
        raise FormulaError("no formula, or too long")
    try:
        tree = ast.parse(formula.replace("×", "*").replace("−", "-"), mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"not a formula: {e}") from None
    steps = []

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise FormulaError(f"unknown name {node.id!r}")
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = _number(ev(node.operand))
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            try:
                v = OPS[type(node.op)](_number(ev(node.left)), _number(ev(node.right)))
            except ZeroDivisionError:
                raise FormulaError("division by zero") from None
            steps.append(v)
            return v
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS
                and not node.keywords):
            try:
                v = float(FUNCS[node.func.id](*[ev(a) for a in node.args]))
            except (TypeError, ValueError, ZeroDivisionError, statistics.StatisticsError) as e:
                raise FormulaError(f"{node.func.id}: {e}") from None
            steps.append(v)
            return v
        raise FormulaError(f"not allowed: {ast.dump(node)[:60]}")

    result = _number(ev(tree))
    if not math.isfinite(result):
        raise FormulaError("not a finite number")
    return result, steps


def _number(v):
    if isinstance(v, list):
        raise FormulaError("a list of rows where one number is needed: use sum(), mean() ...")
    return v


def bind(plan_names, rows):
    """{name: value or [values]} from the model's {name: row index or [row indices]}; FormulaError if a name or an
    index is bad or a row has no number."""
    if not isinstance(plan_names, dict) or not plan_names:
        raise FormulaError("no names")
    out = {}
    for name, ref in plan_names.items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in FUNCS:
            raise FormulaError(f"bad name {name!r}")
        try:
            idx = [int(i) for i in ref] if isinstance(ref, list) else int(ref)
            if any(i < 0 for i in (idx if isinstance(idx, list) else [idx])):
                raise IndexError("negative row")
            vals = [float(rows[i]["value"]) for i in idx] if isinstance(idx, list) else float(rows[idx]["value"])
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise FormulaError(f"{name}: {e}") from None
        if isinstance(vals, list) and not vals:
            raise FormulaError(f"{name}: no rows")
        out[name] = vals
    return out
