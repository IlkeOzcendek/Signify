import streamlit as st

from app_pages.analysis_page import render_analysis_page
from app_pages.verify_page import render_verify_page # Creates the signature verifying page
from app_pages.admin_page import render_admin_page
from app_pages.about_page import render_about_page

from utils.database import load_registered_people
from utils.ui import SIGNIFY_ICON_PATH, icon_data_uri, load_custom_css, show_header
from utils.i18n import t # t = translition, Turkish / English

st.set_page_config ( # Create the application page with those setting down below
    page_title = "Signify",
    page_icon = str(SIGNIFY_ICON_PATH), # favicon
    layout = "wide", # The content uses the entire width of the screen. The margins are reduced / Another alternative: centered
    initial_sidebar_state = "auto", # auto, collapsed or expanded -> How do you want the sidebar to act when you first open the application
) # As a rule of Streamlit you first need to write that before your custom css

load_custom_css()

with st.sidebar: # with: Gives a focus to the sidebar
    st.markdown(
        f"""
        <div class = "sidebar-brand">
            <span class = "brand-mark">
                <img src = "{icon_data_uri()}" alt = "Signify">
            </span>
            <span>Signify</span>
            </div>
        """,
        unsafe_allow_html = True, # Allows writing these in HTML
    )
    st.caption(t("sidebar_caption")) # It adds a small and fainted description below the sidebar -> "Signature verification workspace"

    st.selectbox( # A language selection box
        t("language"), # Translates the heading of the box

        ["tr", "en"], # The options that the program is going to use
        index = 1,
        format_func = lambda value: "Türkçe" if value == "tr" else "English", # format_func = how the options are going to look, lambda = Creates a temporary function for the options and their answers

        key = "language", # It keeps the answer of the "language" in that var.
    )

    page = st.radio(
        t("navigation"),
        ["verify", "admin", "analysis", "about"],

        format_func = lambda value: {
            "verify": t("verify_nav"),
            "admin": t("admin_nav"),
            "analysis": t("analysis_nav"),
            "about": t("about_nav"),
        }
        [value],

        key = "navigation",
    )

    st.markdown('<div class = "sidebar-spacer"></div>', unsafe_allow_html = True) # Used to create a space
    st.caption(t("sidebar_footer"))

if "people" not in st.session_state:
    st.session_state.people = load_registered_people()

if page != "verify":
    show_header()

if page == "verify":
    render_verify_page()

elif page == "admin":
    render_admin_page()

elif page == "analysis":
    render_analysis_page()

elif page == "about":
    render_about_page()
