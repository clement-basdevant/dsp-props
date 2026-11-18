"""First-order mutants of a Python source file, with classic mutation operators.

Operators (Offutt et al. 1996; Jia and Harman 2011), as in mutmut or MutPy:

* AOR  arithmetic operator replacement   + <-> -, * <-> /, // -> /, % -> //, ** -> *
* BOR  bitwise operator replacement      << <-> >>, & <-> |
* ROR  relational operator replacement   < <-> <=, > <-> >=, == <-> !=
* CRP  constant replacement              integer k -> k + 1 and k - 1, float c -> c + 1
* UOI  unary operator deletion           -e -> e
* MCR  min/max call replacement          min <-> max
* SDL  statement deletion                an augmented assignment becomes ``pass``

Each mutant changes exactly one site. Docstrings and the module ``__doc__``
are left alone. ``mutants(path)`` returns a list of ``Mutant`` objects whose
``source`` is the complete mutated module.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

AOR = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Div], ast.Div: [ast.Mult],
       ast.FloorDiv: [ast.Div], ast.Mod: [ast.FloorDiv], ast.Pow: [ast.Mult]}
BOR = {ast.LShift: [ast.RShift], ast.RShift: [ast.LShift], ast.BitAnd: [ast.BitOr], ast.BitOr: [ast.BitAnd]}
ROR = {ast.Lt: [ast.LtE], ast.LtE: [ast.Lt], ast.Gt: [ast.GtE], ast.GtE: [ast.Gt],
       ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq]}


@dataclass
class Mutant:
    id: str
    operator: str
    line: int
    description: str
    source: str


class _Sites(ast.NodeVisitor):
    """Collect (path, operator, replacement) for every mutable site, in source order."""

    def __init__(self):
        self.sites = []
        self.stack = []

    def generic_visit(self, node):
        self.stack.append(node)
        super().generic_visit(node)
        self.stack.pop()

    def visit_BinOp(self, node):
        for repl in AOR.get(type(node.op), []) + BOR.get(type(node.op), []):
            kind = "AOR" if type(node.op) in AOR else "BOR"
            self.sites.append((node, kind, ("op", repl)))
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        for repl in AOR.get(type(node.op), []) + BOR.get(type(node.op), []):
            kind = "AOR" if type(node.op) in AOR else "BOR"
            self.sites.append((node, kind, ("op", repl)))
        self.sites.append((node, "SDL", ("delete", None)))
        self.generic_visit(node)

    def visit_Compare(self, node):
        for i, op in enumerate(node.ops):
            for repl in ROR.get(type(op), []):
                self.sites.append((node, "ROR", ("cmp", (i, repl))))
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub):
            self.sites.append((node, "UOI", ("unwrap", None)))
        self.generic_visit(node)

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return
        if isinstance(v, int):
            for new in (v + 1, v - 1):
                if new >= 0 or v < 0:
                    self.sites.append((node, "CRP", ("value", new)))
        elif isinstance(v, float):
            self.sites.append((node, "CRP", ("value", v + 1.0)))

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
            self.sites.append((node, "MCR", ("func", "max" if node.func.id == "min" else "min")))
        self.generic_visit(node)


def _apply(tree, target, action):
    """Return a mutated deep copy of ``tree`` where ``target`` (an original node) is changed."""
    mapping = {}
    new_tree = copy.deepcopy(tree)
    # walk both trees in parallel to find the copy of ``target``
    for old, new in zip(ast.walk(tree), ast.walk(new_tree)):
        mapping[id(old)] = new
    node = mapping[id(target)]
    kind, arg = action
    if kind == "op":
        node.op = arg()
    elif kind == "cmp":
        i, repl = arg
        node.ops[i] = repl()
    elif kind == "value":
        node.value = arg
    elif kind == "func":
        node.func.id = arg
    elif kind == "unwrap":
        # replace -e by e in its parent
        for parent in ast.walk(new_tree):
            for field, value in ast.iter_fields(parent):
                if value is node:
                    setattr(parent, field, node.operand)
                elif isinstance(value, list):
                    for k, item in enumerate(value):
                        if item is node:
                            value[k] = node.operand
    elif kind == "delete":
        for parent in ast.walk(new_tree):
            for _field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for k, item in enumerate(value):
                        if item is node:
                            value[k] = ast.Pass()
    ast.fix_missing_locations(new_tree)
    return new_tree


def _describe(node, action, source_lines):
    kind, arg = action
    line = source_lines[node.lineno - 1].strip()
    if kind == "op":
        what = f"operator -> {arg.__name__}"
    elif kind == "cmp":
        what = f"comparison -> {arg[1].__name__}"
    elif kind == "value":
        what = f"constant {node.value!r} -> {arg!r}"
    elif kind == "func":
        what = f"{node.func.id}() -> {arg}()"
    elif kind == "unwrap":
        what = "unary minus removed"
    else:
        what = "statement deleted"
    return f"{what} in `{line}`"


def mutants(path) -> list[Mutant]:
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    visitor = _Sites()
    visitor.visit(tree)
    lines = source.splitlines()
    out = []
    stem = path.split("/")[-1].removesuffix(".py")
    for i, (node, op, action) in enumerate(visitor.sites):
        mutated = _apply(tree, node, action)
        try:
            new_source = ast.unparse(mutated)
            compile(new_source, path, "exec")
        except (SyntaxError, ValueError):
            continue
        out.append(Mutant(f"{stem}-{i:03d}", op, node.lineno, _describe(node, action, lines), new_source))
    return out


if __name__ == "__main__":
    import sys
    from collections import Counter

    for p in sys.argv[1:]:
        ms = mutants(p)
        print(p, len(ms), dict(Counter(m.operator for m in ms)))
