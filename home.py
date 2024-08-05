import streamlit as st
import os

root_path = os.path.dirname(__file__)
image_path = os.path.join(root_path,'AI-LAB-LOGO.png')

st.set_page_config(page_title="My App", page_icon= image_path , layout="centered")

st.session_state['login'] = False if 'login' not in st.session_state else True
st.session_state['username'] = "" if 'username' not in st.session_state else st.session_state['username']

def main():

    
    st.sidebar.image(image_path, use_column_width=True)
    st.sidebar.page_link("home.py",label="Home")
    st.sidebar.page_link("pages/sign_up.py",label="Sign Up")
    st.sidebar.page_link("pages/log_in.py",label="Login")
    st.sidebar.page_link("pages/scan.py",label="Scan")
    st.sidebar.page_link("pages/history.py",label="History")

    # Title of the app
    st.title("Welcome to Mental Health Monitoring System.")

    # About Section
    st.markdown("""
    <style>
    .about-section {
        font-family: 'Courier New', Courier, monospace; 
        color: #333333;
        font-size: 18px;
        padding: 10px;
        /*background-color: #f9f9f9;*/
        background-color: #87CEEB;
        border-radius: 10px;
        border: 1px solid #dddddd;
    }
    </style>
    <div class="about-section">
        <h2>About</h2>
        <p>
            Your mental health matters!
        </p>
        <p>
            Prioritize your mental health with our AI-powered emotion tracker. Using advanced facial recognition technology, our app offers real-time insights into your emotional state. Understand your patterns, identify triggers, and take steps to improve your overall well-being. Your mental health journey starts here.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.text("")
    st.text("")

    if st.button("Let's start the journey."):
        st.switch_page("pages/sign_up.py")


    # sky_color = "#87CEEB"
    # # Define custom CSS for button colors
    # button1_color = "#00FF00"  # Green

    # # Create buttons with custom CSS
    # st.markdown(f"""
    # <button style="background-color: {sky_color};" onclick="switch_page('pages/sing_up.py')"; border-radius: 5px;">Let's start the journey</button></a>
    # """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


#///////////////////////////////////////////////////////////////////////////////////////////////////////////

