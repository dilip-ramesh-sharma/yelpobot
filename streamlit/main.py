import os
import requests
import streamlit as st
from jose import JWTError, jwt
import render_analytics_page
import plotly.express as px

from dotenv import load_dotenv

load_dotenv()
# BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
BASE_URL = "http://127.0.0.1:8001"


SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"

def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except:
        return None

def register_user(username, password):
    payload = {"username": username, "password": password}
    response = requests.post(f"{BASE_URL}/register", data=payload)
    return response

def login_user(username, password):
    payload = {"username": username, "password": password}
    response = requests.post(f"{BASE_URL}/login", data=payload)
    return response

def signup():
    st.title("Sign Up")
    username = st.text_input("Enter username")
    password = st.text_input("Enter password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if password == confirm_password:
        if st.button("Sign up"):
            response = register_user(username, password)
            if response.status_code == 200:
                st.success(response.json().get("message"))
    else:
        st.error("Passwords do not match")

def signin():
    st.title("Sign In")
    username = st.text_input("Enter username")
    password = st.text_input("Enter password", type="password")

    if st.button("Sign in"):
        response = login_user(username, password)
        if response.status_code == 200:
            st.success("Sign in successful")
            access_token = response.json().get("access_token")
            print(access_token)
            return access_token
        else:
            st.error("Something went wrong")


# Define the Streamlit pages
pages = {
    "Analytics": render_analytics_page.render_analytics_dashboard,
}


# Define the Streamlit app
def main():
    st.set_page_config(
        page_title="Yelpobot Streamlit Application", 
        layout="wide"
    )
    st.sidebar.title("Welcome to Yelpobot Application")

    # Check if user is signed in
    token = st.session_state.get("token", None)
    user_id = decode_token(token)

    # Render the navigation sidebar
    if user_id is not None:
        selection = st.sidebar.radio("Go to", list(pages.keys()) + ["Log Out"])
    else:
        selection = st.sidebar.radio("Go to", ["Sign In", "Sign Up"])

    # Render the selected page or perform logout
    if selection == "Log Out":
        st.session_state.clear()
        st.sidebar.success("You have successfully logged out!")
        st.experimental_rerun()
    elif selection == "Sign In":
        token = signin()
        if token is not None:
            st.session_state.token = token
            print(token)
            st.experimental_rerun()
    elif selection == "Sign Up":
        signup()
    else:
        page = pages[selection]
        page()


if __name__ == "__main__":
    main()