import streamlit as st
from app.excel_reader import read_excel_file
from app.summarizer import summarize_dataframe_smart
from app.llm_engine import generate_commentary

def launch_ui():
    """
    Launch the Streamlit UI for the Local Excel AI Assistant.
    This function sets up the page, handles file uploads, and displays results.
    """
    st.set_page_config(page_title="Local Excel AI Assistant", layout="wide")
    st.title("📊 Local Excel AI Assistant")
    st.markdown("Upload an Excel file to get insights, summaries, and AI-generated commentary.")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file:
        # Step 1: Read all sheets from the uploaded Excel
        with st.spinner("Reading Excel file..."):
            excel_data = read_excel_file(uploaded_file)

        # Step 2: Let user select a sheet
        sheet_names = list(excel_data.keys())
        selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)

        if selected_sheet:
            df = excel_data[selected_sheet]

            # Step 3: Summarize the selected sheet
            with st.spinner(f"Summarizing data from '{selected_sheet}'..."):
                profile, text_summary = summarize_dataframe_smart(df, selected_sheet)

            # Step 4: Get AI commentary for this sheet
            with st.spinner("Generating AI commentary..."):
                commentary = generate_commentary(text_summary)

            # Step 5: Display outputs
            st.subheader(f"📄 AI Commentary for '{selected_sheet}'")
            st.write(commentary)

            # Optional: Show the dataframe
            with st.expander("🔍 View DataFrame"):
                st.dataframe(df)

            # Optional: Show structured profile
            with st.expander("🗂 Structured Sheet Profile"):
                st.json(profile)
