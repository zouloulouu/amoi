# Streamlit Legacy

This directory keeps the former Streamlit application available as a fallback
while production runs on FastAPI + React.

## Run Locally

From the repository root:

```powershell
python -m venv .venv-streamlit
.\.venv-streamlit\Scripts\Activate.ps1
python -m pip install -r legacy\streamlit\requirements.txt
streamlit run legacy\streamlit\app.py
```

The app still reads shared project resources from the repository root:

- `dictionaries.json`
- `data/clean/CURRENT` when available
- HuggingFace fallback through `HF_TOKEN`

## Secrets

For Streamlit Cloud or local Streamlit secrets, use `secrets.toml.example` as a
template. Never commit a real `secrets.toml`.
