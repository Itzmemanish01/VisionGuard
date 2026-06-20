import sys
import os

# 1. Direct Python to your dashboard folder
sys.path.append(os.path.abspath("./dashboard"))

# 2. Safely read and execute your main application code
with open("dashboard/app.py") as f:
    code = f.read()
exec(code)