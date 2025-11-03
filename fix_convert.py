#!/usr/bin/env python3
"""Fix convert_from_jax.py to handle both 'value' and direct access"""

import re

# Read the file
with open('convert_from_jax.py', 'r') as f:
    content = f.read()

# Replace all ['value'] accesses to use get_weight helper
# Pattern: dump_weights['...']['...'][...]['value']
# We need to be careful not to replace the one in get_weight function itself

lines = content.split('\n')
new_lines = []
in_get_weight_func = False

for i, line in enumerate(lines):
    if 'def get_weight(' in line:
        in_get_weight_func = True
    elif in_get_weight_func and (line.strip().startswith('def ') or (line and not line[0].isspace() and 'def ' in line)):
        in_get_weight_func = False
    
    # Skip modification in get_weight function
    if in_get_weight_func or 'return result[\'value\']' in line or 'embedding_weight = embedding_path[\'value\']' in line:
        new_lines.append(line)
        continue
    
    # Replace ['value'] at the end of dict access chains
    # This pattern catches: ...]['key']['value']
    if "[\'value\']" in line or '["value"]' in line:
        # Remove ['value'] but keep everything else
        line = re.sub(r"\[\'value\'\](?!.*\[\'value\'\])", "", line)
        line = re.sub(r'\["value"\](?!.*\["value"\])', "", line)
    
    new_lines.append(line)

# Write back
with open('convert_from_jax.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("Fixed convert_from_jax.py - removed all ['value'] accesses")
