import json
import os

notebook_path = "experiments/01012026_branching/dosage_sensitivity.ipynb"
code_path = "experiments/01012026_branching/verify_hypothesis.py"

# Read the corrupted file
with open(notebook_path, 'r') as f:
    content = f.read()

# Find the last closing brace which should be the end of the valid JSON
last_brace_index = content.rfind('}')
if last_brace_index == -1:
    print("Error: Could not find closing brace of JSON")
    exit(1)

# Extract valid JSON part
json_content = content[:last_brace_index+1]

# Check if there is trailing garbage (the raw code I appended)
garbage = content[last_brace_index+1:].strip()
if garbage:
    print(f"Found {len(garbage)} characters of trailing garbage. Removing it.")
else:
    print("No trailing garbage found. Proceeding to add cell.")

try:
    notebook = json.loads(json_content)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    # Try to find the *real* end. The file ends with "metadata": {...}, "nbformat": 4, "nbformat_minor": 5}
    # If I appended text, it is after the last } corresponding to the root object.
    # rfind('}') finds the last } in the whole string.
    # If the appended text contains }, rfind will match that.
    # My appended python code contains } (in the print statements or dicts).
    # So I need to be smarter. The notebook ends with `}`.
    # The valid JSON structure should end with `}`.
    # Since I appended text, the *original* last } is somewhere before the end.
    
    # Heuristic: Scan from the beginning and decode? No.
    # Heuristic: The garbage I appended starts with "# ---------------------------------------------------------"
    split_marker = "# ---------------------------------------------------------"
    if split_marker in content:
        print("Found split marker.")
        json_content = content.split(split_marker)[0]
        notebook = json.loads(json_content)
    else:
        print("Could not recover JSON.")
        exit(1)

# Read the code to add
with open(code_path, 'r') as f:
    new_code = f.read()

# Create new cell
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_code.splitlines(keepends=True)
}

# Append cell
notebook['cells'].append(new_cell)

# Write back
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook repaired and updated successfully.")


