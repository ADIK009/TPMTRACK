import pandas as pd
import streamlit as st
from pandasai import PandasAI
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import openai

openai_api_key = "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"  # Replace with your actual OpenAI API key
model = "gpt-3.5-turbo-16k"


def perform_data_analysis(df, prompt):
    llm = openai.OpenAI(api_token=openai_api_key)
    pandas_ai = PandasAI(llm)
    with st.spinner("Generating response..."):
        return pandas_ai.run(df, prompt=prompt)


def generate_ai_summary(df):
    prompt = f"Describe the dataset:\n{df.describe().to_string()}"
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=200  # Adjust the number of tokens based on API limitations and desired output length
    )
    return response['choices'][0]['text']


def main():
    st.title("DATA ANALYSIS")
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("DATA ANALYSIS")
        choice = st.radio("Navigation", ["upload", "Profiling", "Modelling"])
        st.info("This project application helps you build and explore your data.")

    if choice == "upload":
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df.head(7))

            # Generate AI-generated summary
            if st.button("Generate AI Summary"):
                ai_summary = generate_ai_summary(df)
                st.write("AI-generated summary:")
                st.write(ai_summary)

        else:
            st.warning("Please upload a CSV file.")

        prompt = st.text_area("Enter your question:")
        if st.button("Generate"):
            if prompt:
                if uploaded_file is not None:
                    response = perform_data_analysis(df, prompt)
                    st.write(response)
                else:
                    st.warning("Please upload a CSV file first.")
            else:
                st.warning("Please enter a question.")

    elif choice == "Profiling":
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            pr = ProfileReport(df, explorative=True)
            st.write(df)
            st.header("REPORT")
            st_profile_report(pr)

    elif choice == "Modelling":
        st.warning("This feature is not available for CSV file analysis.")


if __name__ == "__main__":
    main()
