import streamlit as st
import pandas as pd
import main  # Import main.py
import ml_predictions
import os
import memory
import deep_learning
import pandas as pd
from groq import Groq
import datavisualization
from data_preprocessing import preprocess_data  # Import preprocessing function

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
# 🔹 Secure API Key

# 🔹 Streamlit UI
st.title("📊 AI Data Analyst")
st.sidebar.header("🔍 Select Task")

# # 🚀 Upload File
uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV File", type=["csv"])

# ✅ Process File
if uploaded_file:
    # Read the uploaded file first
    df = pd.read_csv(uploaded_file)  # Convert file-like object to DataFrame
    
    st.success("✅ File Uploaded Successfully!")

    # Show preview of the uploaded data
    st.write("📊 **Preview of Uploaded Data:**")
    st.write(df.head())  

    # Now process the data
    df_cleaned = main.process_data(df)  # Process the already-read DataFrame
    st.success("✅ File Processed! Ready for Analysis.")

    # st.write(uploaded_file.head())  # Display first few rows of cleaned data

    # Get AI-generated analysis based on the cleaned data
    st.write(main.generate_ai_analysis(df_cleaned))

    # 🎯 Select Task
    # 🚀 Select Task
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Select...", "Train ML Model", "DL prep", "Data Visualize", "Chat with the model"]
)

# 🚀 Perform Selected Task
if option == "Train ML Model":
    st.subheader("🚀 Train Machine Learning Model")
    results = ml_predictions.analyze_ml_patterns(df_cleaned)
    # st.write("Model Trained!")
    summary_text = df_cleaned.describe().to_string()
    # st.write(f"** **\n{model}")
    results =  ' | '.join(f"{key}: {str(value)}" for key, value in results.items())
    print(results)
     # # response = main.client(df_cleaned, user_query)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": summary_text},
                  {"role": "system", "content": results },
                  
                  {"role": "user", "content": "based on the  give summary_text of dataset different model results find and analyse the patterns and insights and give judgement to every result present in the given results"}],
        model="mixtral-8x7b-32768",
        stream=False,
    )
    response =  chat_completion.choices[0].message.content 
    memory.main(query="ML Model Analysis", summary=f"{results} | AI Response: {response}")
 # Return AI response
    # formatted_response = f"""
    # <div style='padding:15px; border-radius:10px; background-color:#f5f5f5; border-left: 5px solid #FF5722;'>
    # <h3 style='color:#E91E63;'>🧠 AI Analysis & Insights:</h3>
    # <p>{response}</p>
    # </div>
    
# Display AI response in Streamlit with formatting
    # st.markdown(formatted_response, unsafe_allow_html=True)
    st.write(f"🧠 {response}")

elif option == "DL prep" :
    st.subheader("🚀Neural Network Results")
    results = deep_learning.analyze_deep_learning_patterns(df_cleaned)
    summary_text = df_cleaned.describe().to_string()
    # st.write(f"** **\n{model}")
    results =  ' | '.join(f"{key}: {str(value)}" for key, value in results.items())
    print(results)
     # # response = main.client(df_cleaned, user_query)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": summary_text},
                  {"role": "system", "content": results },
                  
                  {"role": "user", "content": "based on the  give summary_text of dataset different deep learning models results find and analyse the patterns and insights and give judgement to every result present in the given results"}],
        model="mixtral-8x7b-32768",
        stream=False,
    )
    response =  chat_completion.choices[0].message.content
    memory.main(query="DL nueral network Analysis", summary=f"{results,response} | AI Response: {response}")

    st.write(response)  # Return AI response
    
elif option == "Data Visualize":
    # st.subheader("📊 Data Visualizations")
    datavisualization.generate_all_plots(df_cleaned)
    # st.write("Model Trained!")
    # st.write(f"** **\n{model}")
    # for key, value in results.items():
    #     st.write(f"### {key}:")
    #     if isinstance(value, pd.DataFrame):
    #         st.dataframe(value)  # Display dataframe results, such as clustering or correlation matrix
    #     elif isinstance(value, pd.Series):
    #         st.write(value)  # Display series results, such as feature importance or regression metrics
    #     else:
    #         st.write(value)
elif option == "Chat with the model":



    st.sidebar.subheader("💬 Chat with the Model")
    user_query = st.sidebar.text_input("Ask something about the data:")
    if user_query:
        summary_text = df_cleaned.describe().to_string()
        response = memory.main(query=user_query,summary=summary_text)
        # # response = main.client(df_cleaned, user_query)
        # chat_completion = client.chat.completions.create(
        #     messages=[{"role": "system", "content": summary_text},
        #               {"role": "user", "content": user_query}],
        #     model="mixtral-8x7b-32768",
        #     stream=False,
        # )

        # response =  chat_completion.choices[0].message.content  # Return AI response

        st.sidebar.write(f"🧠 Model Response: {response}")

