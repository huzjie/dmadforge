"""A minimal YAML subset parser (zero-dependency fallback).

Supports the subset our configs actually use: nested mappings, block lists
(`- item` and `- key: value`), inline lists `[a, b]`, scalars (str/int/float/
bool/null), and inline comments after ` # `. Falls back gracefully when the
real PyYAML is not installed.
"""
import re


def _clean(line):
    line = line.rstrip()
    if not line:
        return line
    # strip inline comments (not inside quotes)
    in_q = None
    out = []
    i = 0
    while i < len(line):
        ch = line[i]
        if in_q:
            out.append(ch)
            if ch == in_q:
                in_q = None
        elif ch in ("\'", '"'):
            in_q = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in (" ", "\t")):
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _scalar(s):
    s = s.strip()
    if s == "" or s == "null" or s == "~":
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    m = re.match(r"^[\[\{]\]$", s)
    if m:
        return s
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_scalar(x) for x in _split_top(inner)]
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        if re.match(r"^[-+]?\d+$", s):
            return int(s)
        return float(s)
    except ValueError:
        return s


def _split_top(s):
    parts = []
    depth = 0
    cur = ""
    in_q = None
    for ch in s:
        if in_q:
            cur += ch
            if ch == in_q:
                in_q = None
        elif ch in ("\'", '"'):
            in_q = ch
            cur += ch
        elif ch in "[{(":
            depth += 1
            cur += ch
        elif ch in "]})":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def parse(text):
    lines = text.splitlines()
    # strip document markers
    lines = [ln for ln in lines if ln.strip() not in ("---", "...")]
    root = {}
    stack = [( -1, root, None, None)]  # (indent, container, parent, key)

    def container_of(indent):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        return stack[-1]

    for raw in lines:
        line = _clean(raw)
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.startswith("- "):
            item = content[2:].strip()
            top = container_of(indent)
            container = top[1]
            if ":" in item and not item.startswith("[") and not item.startswith("{"):
                k, v = item.split(":", 1)
                k = k.strip()
                if isinstance(container, dict):
                    lst = container.get(top[3])
                    if not isinstance(lst, list):
                        lst = []
                        container[top[3]] = lst
                    d = {}
                    d[_scalar(k)] = _scalar(v) if v.strip() else None
                    lst.append(d)
                else:
                    d = {_scalar(k): _scalar(v) if v.strip() else None}
                    container.append(d)
            else:
                if isinstance(container, dict):
                    lst = container.get(top[3])
                    if not isinstance(lst, list):
                        lst = []
                        container[top[3]] = lst
                    lst.append(_scalar(item))
                else:
                    container.append(_scalar(item))
            continue
        if ":" not in content:
            continue
        key, _, val = content.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            # parent of a nested block
            node = {}
            if indent > stack[-1][0]:
                parent = stack[-1][1]
                parent[_scalar(key)] = node
                stack.append((indent, node, parent, _scalar(key)))
            else:
                top = container_of(indent)
                top[1][_scalar(key)] = node
                stack.append((indent, node, top[1], _scalar(key)))
        else:
            container = container_of(indent)[1]
            container[_scalar(key)] = _scalar(val)
    return root
