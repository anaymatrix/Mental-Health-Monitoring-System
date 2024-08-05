
import streamlit as st
import pandas as pd
import os
import base64

root_path = os.path.dirname(__file__)
image_path = os.path.join(root_path,'AI-LAB-LOGO.png')

st.sidebar.image(image_path, use_column_width=True)
st.sidebar.page_link("home.py",label="Home")
st.sidebar.page_link("pages/sign_up.py",label="Sign Up")
st.sidebar.page_link("pages/log_in.py",label="Login")
st.sidebar.page_link("pages/scan.py",label="Scan")
st.sidebar.page_link("pages/history.py",label="History")




def main():
    st.title("Track your all records.")

    # Get the list of all CSV files
    all_csv_files = [f for f in os.listdir('pages/history') if f.endswith('.csv')]

    # Get unique usernames from file names
    # usernames = set(f.split('_')[0] for f in all_csv_files)

    # Select a username
    # selected_username = st.selectbox("Select a username", list(usernames))

    selected_username = st.session_state['username']

    # Filter CSV files for the selected username
    user_csv_files = [f for f in all_csv_files if f.startswith(selected_username)]

    # Display the list of CSV files for the selected user
    selected_file = st.selectbox("Select a CSV file", user_csv_files)

    if selected_file:
        # Read the selected CSV file
        df = pd.read_csv(os.path.join(root_path,'history/', selected_file))

        # Display the CSV content
        st.dataframe(df)

        # Download the CSV file
        def get_table_download_link(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings
            href = f'<a href="data:file/csv;base64,{b64}">download="download.csv">Download CSV</a>'
            return href

        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

if __name__ == "__main__":
    if "login" not in st.session_state:
        st.switch_page(os.getcwd()+"/home.py")
    if st.session_state['login']:
        main()
    else:
        st.warning('Please Login first.', icon="⚠️")
        if st.button("Login"):
            st.switch_page("pages/log_in.py")