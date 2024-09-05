# Q&A Chatbot
#from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap

#from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load OpenAI model and get respones

def get_gemini_response(input_text):
     # Create a GenerativeModel instance with 'gemini-pro' as the model type
    llm = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
    )
    # Generate content based on the input text
    response = llm.generate_content(input_text)
    # Return the generated text
    return response.text

##initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")

st.header("Gemini Clone")

input=st.text_input("Input: ",key="input")


submit=st.button("Ask the question")

## If ask button is clicked

if submit:
    
    response=get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)
