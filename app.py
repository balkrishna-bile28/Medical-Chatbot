from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
# from src.promopt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from openai.error import RateLimitError, OpenAIError
from openai import OpenAIError, RateLimitError
from dotenv import load_dotenv
from src.promopt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_embeddings()

index_name = "med-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])

def chat():
    msg = request.form["msg"]
    if not msg.strip():
        return "Please enter a valid message."

    print("User input:", msg)

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        print("Response:", answer)
        return str(answer)

    except RateLimitError:
        print("Rate limit exceeded.")
        return "Sorry, the service is currently unavailable due to usage limits. Please try again later."

    except OpenAIError as e:
        print("OpenAI API error:", str(e))
        return "There was a problem with the AI service. Please try again later."

    except Exception as e:
        print("Unexpected error:", str(e))
        return "An unexpected error occurred. Please try again later."



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)