import streamlit as st
from pathlib import Path

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Arraj2611"
PAGE_ICON = "./assets/logo.png"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"

with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

#---- PAGE SETUP ----
profile_page = st.Page(
    page='views/profile_page.py',
    title='About Me',
    icon='🏡',
    default=True,
)
chatbot = st.Page(
    page='views/chatbot.py',
    title='Chat-with-Mi',
    icon='🤖',
)
dog_breeds = st.Page(
    page='views/dog_breeds.py',
    title='Dog-Breed Identification',
    icon='🐶',
)

#----PAGE NAV ----
pg = st.navigation(
    {
        'Info': [profile_page],
        'Projects': [chatbot, dog_breeds],
    }
)

# ---- Sidebar-mods ----
st.logo('assets/image.png')
st.markdown(
    '''
    <style>
    img[data-testid="stLogo"] {
            height: 3.5rem;
}
</style>
    ''', unsafe_allow_html=True)
st.sidebar.text('Made with ⚡ by Rajeev')
#----NAV RUN
pg.run()