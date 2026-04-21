import os
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel


load_dotenv()
project=os.getenv('GOOGLE_CLOUD_PROJECT')

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    vertexai=True,
    project=project
)
   
