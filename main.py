import os
import streamlit as st 
import pandas as pd
from groq import Groq
from data_preprocessing import preprocess_data  # Import preprocessing function

# 🔹 Secure API Key
# os.environ["GROQ_API_KEY"] = "blahblah"

# 🔹 Initialize Groq Client
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# ✅ Function to Process Data
def process_data(uploaded_file):
    """Handles file processing when called from app.py."""
    try:
        # ✅ Correct way to read uploaded file
        df = pd.read_csv(uploaded_file)  # Convert file to DataFrame directly
        df_clean = preprocess_data(df)   # Process the data (pass DataFrame)
        print(type(df_clean), "is the cleaned data")  # Preprocess data
        return df_clean  # Return cleaned DataFrame
    except Exception as e:
        return f"❌ Error in processing file: {e}"

# ✅ AI Analysis Function
def generate_ai_analysis(df_clean):
    """Generates AI-based insights on the cleaned dataset."""
    summary_text = df_clean.describe().to_string()  # Get summary stats

    analysis_prompt = f"""
    You are an expert data analyst. Analyze the dataset in detail.

    ### **1️⃣ Key Summary Statistics**
    {summary_text}

    ### **2️⃣ Hidden Trends & Patterns**
    - Identify seasonal trends, correlations, or anomalies.
    - Find patterns related to time, sales, or other variables.

    ### **3️⃣ Strengths & Weaknesses**
    - Highlight positive trends (e.g., high growth areas).
    - Identify weaknesses (e.g., declining sales, high variance).

    ### **4️⃣ Real-World Business Insights**
    - Explain the business impact of the findings.
    - Suggest actionable recommendations for decision-making.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI data analyst."},
                  {"role": "user", "content": analysis_prompt}],
        model="mixtral-8x7b-32768",
        stream=False,
    )

    return chat_completion.choices[0].message.content  # Return AI response
