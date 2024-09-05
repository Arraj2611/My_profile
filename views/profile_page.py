import streamlit as st
from pathlib import Path
from forms.contact_form import contact_form

SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/rajeevaken/",
    "GitHub": "https://github.com/Arraj2611",
    "X": "https://x.com/AkenRajeev",
}

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
resume_file = current_dir / "assets" / "Resume.pdf"

with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()


@st.dialog("Contact Me")
def show_contact_form():
    contact_form()

# -- Hero section --
col1, col2 = st.columns(2, gap='small', vertical_alignment='center')
with col1:
    st.image('./assets/profile-pic.png', width=230)
with col2:
    st.title('Rajeev Aken', anchor=False)
    st.write(
        "A machine learning enthusiast with a desire to explore the evergrowing forest of technology. I am eager to apply theoretical knowledge in the industry and hone my skills and knowledge."
    )
    if st.button('✉️ Contact Me'):
        show_contact_form()
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )

# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")


# --- About Me ---
st.write("\n")
st.subheader("About Me", anchor=False)
st.write(
    """
    - Hands-on experience and knowledge in Python
    - Good understanding of theoretical knowledge about ML and their respective applications
    - Excellent team-player and displaying a strong sense of initiative on tasks
    """
)

# --- Education & QUALIFICATIONS ---
st.write("\n")
st.subheader("Education", anchor=False)
st.write(
    """
    - **B.Tech in Electronics and Computer Engineering**, *WIT Solapur*
    - **Higher Secondary Education**, *Walchand college of arts and science*, *Solapur*
    """
)

# --- Experience ---
st.write("\n")
st.subheader("Experience", anchor=False)
st.write(
    """
    ##### **SIH 2023**
    *Problem Statement* : Sentiment Analysis of Incoming calls on a Helpdesk
    - Worked on Sentiment analysis and Emotion detection using LSTM models
    - Utilised capabilities of TensorFlow library employing Deep Learning Techniques

    ##### **Software Development Intern**
    *Shri Sai Tech LLC, Solapur*
    - Worked on Flask RESTful APIs for application backend
    - Worked on AI Agents using the Microsoft AutoGen and Agency Swarm Frameworks
    - Worked on Android app development using React Native
    """
)
# --- SKILLS ---
st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write(
    """
    - Programming: Python, SQL, C++
    - Modules: TensorFlow, keras, LangChain, Scikit-learn
    - Technologies: Git, Github, Streamlit, Flask
    - Databases: MySQL, PostgreSQL
    """
)

st.write("\n")
st.subheader("Soft Skills", anchor=False)
st.write(
    """
    - Soft Skills: Problem Solving, Teamwork, Time management, Abstraction
    - Hobbies: Playing sports like football, basketball, volleyball, cricket, watching movies, playing video games and listening to music.
    - languages: English, Marathi, Hindi, Telugu, German(learning), Japanese(learning)
    """
)