import streamlit as st
import requests

BACKEND = "http://localhost:8000"

st.title("College Assistant — MVP")

uploaded = st.file_uploader("Upload PDF", type="pdf")
if uploaded:
    resp = requests.post(f"{BACKEND}/upload", files={"file": uploaded})
    st.write(resp.json())

q = st.text_input("Ask a question")
if st.button("Ask") and q:
    r = requests.get(f"{BACKEND}/ask", params={"q": q})
    st.write(f"Status code: {r.status_code}")
    st.write(f"Response text: {repr(r.text)}")  # Show raw response

    try:
        data = r.json()
        st.subheader("Answer")
        st.write(data.get("answer", "No answer key in response"))
        st.subheader("Sources")
        st.write(data.get("sources", "No sources key in response"))
    except requests.exceptions.JSONDecodeError:
        st.error("Failed to parse JSON from backend response.")

