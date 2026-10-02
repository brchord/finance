"""
Local UI for the Monte Carlo retirement engine:

    streamlit run app.py

The UI never runs simulation code itself: it launches monte_carlo.py
--job-dir as a detached subprocess and only reads the files it writes.
See doc/plans/UI Design.md.
"""

import streamlit as st

from planner import ui

st.set_page_config(page_title="Retirement planner", layout="wide")

pages = st.navigation([
    st.Page("pages/explorer.py", title="Explorer", default=True),
    st.Page("pages/cell.py", title="Cell detail"),
    st.Page("pages/history.py", title="History"),
])
ui.sidebar()
pages.run()
