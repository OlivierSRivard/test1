import os
import streamlit as st

def get_secret(key, default=""):
    # Prefer environment variables (Render), then Streamlit secrets if present
    val = os.environ.get(key)
    if val is not None:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default
import os
import streamlit as st

def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)
# Home.py
import os
import streamlit as st

st.set_page_config(page_title="Home", layout="wide")
APP_VERSION = "home-2025-09-25e+gate"

# ---------- helpers ----------
def _hide_sidebar_nav():
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display:none !important;}
            [data-testid="stSidebarNav"] {display:none !important;}
            [data-testid="collapsedControl"] {display:none !important;}
        </style>
    """, unsafe_allow_html=True)

def get_secret(key: str) -> str | None:
    # Prefer Streamlit secrets (local dev), then environment (Render)
    if key in st.secrets:
        return str(st.secrets[key])
    return os.getenv(key)

# ---------- credentials (secrets only) ----------
BASIC_USER = (get_secret("BASIC_USER") or "").strip().lower()
BASIC_PASS = get_secret("BASIC_PASS") or ""

def auth_ok(email: str, password: str) -> bool:
    return (email.strip().lower() == BASIC_USER) and (password == BASIC_PASS)

# ---------- login gate ----------
if not st.session_state.get("is_authed"):
    _hide_sidebar_nav()

    st.title("Sign in")
    with st.form("login", clear_on_submit=False):
        email = st.text_input("Email", placeholder="you@company.com")
        pwd   = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if not BASIC_USER or not BASIC_PASS:
            st.error("Login is not configured. Contact the admin.")
        elif auth_ok(email, pwd):
            st.session_state["is_authed"] = True
            st.rerun()
        else:
            st.error("Invalid email or password.")
    st.stop()  # block anything else until authed

# ---------- main (only after auth) ----------
with st.sidebar:
    st.write(f"App version: {APP_VERSION}")
    if st.button("🔄 Clear caches & rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.title("Home")
st.write("Use the links below to open pages:")

st.page_link("pages/01_Company_Search.py", label="Company Search", icon="🔎")
st.page_link("pages/02_ETF_Mean_Reversion.py", label="ETF Mean Reversion", icon="📈")
st.page_link("pages/03_Margin__Optimization.py", label="Margin Optimization", icon="⚖️")


