from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
import qdrant_client
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st

st.set_page_config(page_title="Chat with the Chat Bot",
                   page_icon="🤖",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
os.environ["COHERE_API_KEY"] = st.secrets.cohere_key
QDRANT_HOST = st.secrets.QDRANT_HOST
QDRANT_API_KEY = st.secrets.QDRANT_API_KEY

st.title("Welcome To GSM infoBot")

if "messages" not in st.session_state.keys():
  # Initialize the chat messages history
  st.session_state.messages = [{
      "role":
      "assistant",
      "content":
      "Need Info? Ask Me Questions about GSM Mall's Features"
  }]

template = """
Keep the response short, give response according to user question. \
The response should be under 15 words. \
Response should be from the data source. \
Respect the time of the user. \
Try to fit in emojis possible in the response. \ 
Encourage users to visit the store without being pushy. \
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

memory = ConversationBufferWindowMemory(k=5,
                                        memory_key="history",
                                        input_key="question")


@st.cache_resource(show_spinner=False)
def load_data():
  with st.spinner(
      text=
      "Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."
  ):

    client = qdrant_client.QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )

    embeddings = CohereEmbeddings(model="embed-english-v2.0")

    vector_store = Qdrant(client=client,
                          collection_name="my_documents",
                          embeddings=embeddings)
    retriver = vector_store.as_retriever(search_type="mmr")
    return retriver


index = load_data()

chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

if "chat_engine" not in st.session_state.keys():
  # Initialize the chat engine
  st.session_state.chat_engine = RetrievalQA.from_chain_type(
      chat,
      chain_type='stuff',
      retriever=index,
      chain_type_kwargs={
          "prompt": prompt,
          "memory": memory
      })

if prompt := st.chat_input("Your question"):
  # Prompt for user input and save to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
  # Display the prior chat messages
  with st.chat_message(message["role"]):
    st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      response = st.session_state.chat_engine.run(prompt)
      st.write(response['answer'])
      message = {"role": "assistant", "content": response['answer']}
      st.session_state.messages.append(
          message)  # Add response to message history
