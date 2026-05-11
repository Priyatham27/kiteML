"""Patch result.py to replace print() with safe_print() for Unicode safety."""

filepath = "kiteml/output/result.py"

with open(filepath, encoding="utf-8") as f:
    content = f.read()

# Replace indented print( calls with safe_print(
# This targets actual code lines, not docstrings
lines = content.split("\n")
new_lines = []
in_docstring = False

for line in lines:
    stripped = line.lstrip()

    # Track triple-quoted strings (simple heuristic)
    triple_count = line.count('"""')
    if triple_count % 2 == 1:
        in_docstring = not in_docstring

    if not in_docstring and stripped.startswith("print(") and "safe_print" not in line:
        line = line.replace("print(", "safe_print(", 1)

    new_lines.append(line)

content = "\n".join(new_lines)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("Done - patched result.py")
