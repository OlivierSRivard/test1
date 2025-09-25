Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 1) Create venv if missing (uses the Python Launcher "py")
if (!(Test-Path ".\.venv")) {
  py -3 -m venv .venv
}

# 2) Upgrade pip and install requirements (if present)
.\.venv\Scripts\python.exe -m pip install --upgrade pip
if (Test-Path ".\requirements.txt") {
  .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
}

# 3) Sanity check entry file
if (!(Test-Path ".\Home.py")) {
  Write-Error "Home.py not found. Update scripts or fix path."
}

# 4) Run Streamlit (quotes handle spaces in your OneDrive path)
.\.venv\Scripts\python.exe -m streamlit run ".\Home.py"
