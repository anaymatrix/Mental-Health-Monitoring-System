import streamlit as st
import json
import os
import time


root_path = os.path.dirname(__file__)
image_path = os.path.join(root_path,'AI-LAB-LOGO.png')
st.sidebar.image(image_path, use_column_width=True)
st.sidebar.page_link("home.py",label="Home")
st.sidebar.page_link("pages/sign_up.py",label="Sign Up")
st.sidebar.page_link("pages/log_in.py",label="Login")
st.sidebar.page_link("pages/scan.py",label="Scan")
st.sidebar.page_link("pages/history.py",label="History")



class LoginApp:
    def __init__(self, data_file='user_data.json'):
        self.username = ""
        self.password = ""
        self.data_file = data_file
        self.load_user_data()

    def load_user_data(self):
        """Load user data from a JSON file."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.user_data = json.load(f)
        else:
            self.user_data = {}

    def display_title(self):
        """Display the title of the app."""
        st.title("Login Page")

    def display_login_form(self):
        """Display the login form."""
        with st.form(key="login_form"):
            self.username = st.text_input("Username", key="login_username")
            self.password = st.text_input("Password", type="password", key="login_password")
            login_button = st.form_submit_button(label="Log In")
        
        st.text("New user should Sign Up first.")

        if st.button("Signup"):
            st.switch_page("pages/sign_up.py")

        return login_button

    def validate_login(self):
        """Validate login credentials."""
        if self.username in self.user_data and self.user_data[self.username]['password'] == self.password:
            return True
        return False

    def display_login_message(self, message, is_success):
        """Display a login message on the UI."""
        if is_success:
            st.success(message)
        else:
            st.error(message)

    def run(self):
        """Run the app."""
        self.display_title()
        
        # Login section
        login_button = self.display_login_form()
        if login_button:
            if self.validate_login():
                st.session_state['login'] = True
                st.session_state['username'] = self.username
                message = f"Welcome back, {self.user_data[self.username]['name']}!"
                self.display_login_message(message, True)
                time.sleep(1.5)
                st.switch_page('pages/scan.py')
            else:
                self.display_login_message("Invalid username or password.", False)

if __name__ == "__main__":
    app = LoginApp()
    app.run()
