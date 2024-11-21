import modal
import streamlit as st
stub = modal.Stub("streamlit")

@stub.function()
def streamlit_app():
    st.write('hello world')

@stub.local_entrypoint()
def main():
    streamlit_app.remote()