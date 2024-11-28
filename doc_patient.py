# -*- coding: utf-8 -*-
"""
Doctor_Patient_Conversation_Analysis
"""

import streamlit as st
import openai  # For GPT-4 (OpenAI API)
# Assuming you have access to Groq's SDK for running models (you would need to import Groq SDK in a different context)
from groq import Groq

# Function to analyze transcription using GPT-4
def analyze_transcription_gpt4(transcription):
    """
    Sends the transcription to GPT-4 for medical analysis.
    Args:
        transcription (str): Text from the transcription.
    Returns:
        str: GPT-4's analysis.
    """
    prompt = f"""
    The following is a conversation between a doctor and a patient:
    {transcription}

    Based on this conversation, provide:
    1. A possible prognosis for the patient.
    2. A detailed diagnosis of the condition.
    3. Medication recommendations or treatments for the patient.
    """
    print("Sending transcription to GPT-4 for analysis...")

    try:
        # Updated code to use the new OpenAI client interface
        client = Groq(
        api_key= st.secrets["API_KEY"]
        )
        response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a medical assistant AI with expertise in prognosis, diagnosis, and medication recommendations."},
            {"role": "user", "content": prompt}
        ]
        )
        analysis = response['choices'][0]['message']['content']
        print("GPT-4 analysis received.")
        return analysis
    except Exception as e:
        print("Error during GPT-4 analysis:", str(e))
        return f"An error occurred during GPT-4 analysis: {str(e)}"

# Streamlit App Setup
st.title("Doctor-Patient Conversation Analysis")
st.write("Enter a conversation between a doctor and a patient to analyze prognosis, diagnosis, and recommendations.")

# Text input for the conversation
st.subheader("Input Doctor-Patient Conversation")
conversation = st.text_area(
    "Paste the conversation here:",
    placeholder="e.g., Doctor: Hello, how are you feeling today? Patient: I've had a fever for three days...",
    height=200
)

# Analyze button
if st.button("Analyze Conversation"):
    if conversation.strip():
        # Step 1: Analyze the conversation with GPT-4
        with st.spinner("Analyzing the conversation..."):
            analysis = analyze_transcription_gpt4(conversation)

        # Step 2: Display the medical analysis
        st.subheader("Medical Analysis")
        st.write(analysis)

        # Step 3: Provide download options for results
        st.download_button(
            label="Download Medical Analysis",
            data=analysis,
            file_name="medical_analysis.txt",
            mime="text/plain"
        )
    else:
        st.error("Please enter a conversation before analyzing.")

# Footer
st.write("---")
st.write("Powered by OpenAI's GPT-4")
