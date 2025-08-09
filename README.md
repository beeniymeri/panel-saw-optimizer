
# Panel Saw Cutting Optimizer (Streamlit)

A Streamlit web app for panel saw cutting optimization with grain-direction, material/thickness matching, remnants library, and exports (CSV/SVG/Excel/Opal2070/Panhans NCR).

## Deploy on Streamlit Community Cloud (free)

1. Create a new **public GitHub repo** and upload these files from this folder:
   - `panel_saw_streamlit_app.py`
   - `requirements.txt`
   - `runtime.txt`
   - `.streamlit/config.toml`
   - `panel_saw_streamlit_app.parts_example.csv`
   - `panel_saw_streamlit_app.sheets_example.csv`
   - `icon.ico` (optional)

2. Go to https://share.streamlit.io and click **New app**.
   - Select your GitHub repo/branch.
   - **Main file:** `panel_saw_streamlit_app.py`
   - Click **Deploy**.

3. After it builds, you’ll get a URL like `https://your-app.streamlit.app`.
   - You can embed this URL on any site via `<iframe>`.

## Notes
- **Python version:** chosen via `runtime.txt` (3.10).  
- **File writes:** temporary only (ephemeral); downloads are served via `st.download_button`.
- If you see a memory error, reduce input sizes or disable Excel export.
- If you need secrets (not required here), add them in Streamlit Cloud’s **Secrets** tab.
