# import streamlit as st

# class SignUpApp:
#     def __init__(self):
#         self.name = ""
#         self.username = ""
#         self.email = ""
#         self.password = ""
#         self.confirm_password = ""
        
#     def display_title(self):
#         """Display the title of the app."""
#         st.title("Sign Up")

#     def display_form(self):
#         """Display the sign-up form."""
#         with st.form(key="signup_form"):
#             self.name = st.text_input("Name")
#             self.username = st.text_input("Username")
#             self.email = st.text_input("Email")
#             self.password = st.text_input("Password", type="password")
#             self.confirm_password = st.text_input("Confirm Password", type="password")
#             submit_button = st.form_submit_button(label="Sign Up")
        
#         return submit_button

#     def validate_form(self):
#         """Validate form inputs and return a message."""
#         if not (self.name and self.username and self.email and self.password and self.confirm_password):
#             return "Please fill in all the fields."
#         if self.password != self.confirm_password:
#             return "Passwords do not match."
#         return f"Welcome, {self.name}! You have successfully signed up with the username: {self.username} and email: {self.email}"

#     def display_message(self, message, is_success):
#         """Display a message on the UI."""
#         if is_success:
#             st.success(message)
#         else:
#             st.error(message)

#     def run(self):
#         """Run the app."""
#         self.display_title()
#         submit_button = self.display_form()

#         if submit_button:
#             is_success = self.name and self.username and self.email and self.password and self.confirm_password and (self.password == self.confirm_password)
#             message = self.validate_form()
#             self.display_message(message, is_success)

# if __name__ == "__main__":
#     app = SignUpApp()
#     app.run()



#///////////////////////////////////////////////////////////////////////////////////////////////////////////






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

class SignUpApp:
    def __init__(self, data_file='user_data.json'):
        self.name = ""
        self.username = ""
        self.email = ""
        self.password = ""
        self.confirm_password = ""
        self.data_file = data_file
        self.load_user_data()

    def load_user_data(self):
        """Load user data from a JSON file."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.user_data = json.load(f)
        else:
            self.user_data = {}
 
    def save_user_data(self):
        """Save user data to a JSON file."""
        with open(self.data_file, 'w') as f:
            json.dump(self.user_data, f, indent=4)

    def display_title(self):
        """Display the title of the app."""
        st.title("Sign Up")

    def display_form(self):
        """Display the sign-up form."""
        with st.form(key="signup_form"):
            self.name = st.text_input("Name")
            self.username = st.text_input("Username")
            self.email = st.text_input("Email")
            self.password = st.text_input("Password", type="password")
            self.confirm_password = st.text_input("Confirm Password", type="password")
            submit_button = st.form_submit_button(label="Sign Up")

        st.text("Existing users should directly Login here.")

        if st.button("Login"):
            st.switch_page("pages/log_in.py")
        
        return submit_button

    def validate_form(self):
        """Validate form inputs and return a message."""
        if not (self.name and self.username and self.email and self.password and self.confirm_password):
            return "Please fill in all the fields."
        if self.username in self.user_data:
            return "Username already exists. Please choose a different username."
        if self.password != self.confirm_password:
            return "Passwords do not match."
        return None

    def register_user(self):
        """Register the user and save their details."""
        self.user_data[self.username] = {
            'name': self.name,
            'email': self.email,
            'password': self.password  # Note: Passwords should be hashed in a real application
        }
        self.save_user_data()

    def display_message(self, message, is_success):
        """Display a message on the UI."""
        if is_success:
            st.success(message)
        else:
            st.error(message)

    def run(self):
        """Run the app."""

        self.display_title()
        submit_button = self.display_form()

        if submit_button:
            validation_message = self.validate_form()
            if validation_message is None:
                self.register_user()
                message = f"Welcome, {self.name}! You have successfully signed up with the username: {self.username} and email: {self.email}"
                self.display_message(message, True)
                st.session_state['login'] = True
                time.sleep(1.5)
                st.switch_page('pages/scan.py')
            else:
                self.display_message(validation_message, False)

if __name__ == "__main__":
    app = SignUpApp()
    app.run()



