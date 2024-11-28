# -*- coding: utf-8 -*-
"""Doc_Patient.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NlBLuvSLCTOj6ETICnRd3tHYXt5_x3LW
"""

# -*- coding: utf-8 -*-
"""Doctor_Patient_Conversation_TextInput"""

import streamlit as st
from groq import Groq

# Function to analyze transcription using GPT-4
def analyze_conversation(conversation_text):
    """
    Sends the provided text to GPT-4 for medical analysis.
    Args:
        conversation_text (str): Text of the doctor-patient conversation.
    Returns:
        str: GPT-4's analysis.
    """
    prompt = f"""
    The following is a conversation between a doctor and a patient:
    {conversation_text}

    Based on this conversation, provide:
    1. A possible prognosis for the patient.
    2. A detailed diagnosis of the condition.
    3. Medication recommendations or treatments for the patient.
    """

    print("Sending conversation to GPT-4 for analysis...")

    try:
        # Updated code to use the new OpenAI client interface
        client = Groq(
            api_key=st.secrets["API_KEY"]
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
st.write("Paste a doctor-patient conversation below to analyze prognosis, diagnosis, and recommendations.")

# Text input for the conversation
conversation_text = st.text_area("Enter the doctor-patient conversation here:", height=300)

if st.button("Analyze Conversation"):
    if conversation_text.strip():
        # Step 1: Analyze the conversation
        with st.spinner("Analyzing the conversation..."):
            analysis = analyze_conversation(conversation_text)

        # Display the medical analysis
        st.subheader("Medical Analysis:")
        st.write(analysis)

        # Step 2: Provide download options for results
        st.download_button(
            label="Download Medical Analysis",
            data=analysis,
            file_name="medical_analysis.txt",
            mime="text/plain"
        )
    else:
        st.error("Please enter a conversation before clicking 'Analyze Conversation'.")