# -*- coding: utf-8 -*-
"""Entry point for the Futures dashboard.

The two pages are separate scripts rather than two tabs of one script on
purpose. `oi_progression.py` is built around a single commodity — its
top-level selector drives module-level state that every one of its tabs reads
— while the seasonals page works on a basket spanning several markets at once,
which has no single commodity to select. Keeping them as pages also means
Streamlit only executes the page being looked at, so a widget on one no longer
re-runs the other's band computations.

Streamlit Cloud must point at THIS file, not at oi_progression.py.
"""
import streamlit as st

st.set_page_config(page_title="Futures Dashboard", page_icon="📈", layout="wide")

st.navigation([
    st.Page("oi_progression.py", title="OI Progression", default=True),
    st.Page("oi_seasonals.py", title="Deferred OI Seasonals"),
]).run()
