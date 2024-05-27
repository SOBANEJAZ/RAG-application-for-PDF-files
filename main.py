# Importing requred libraries of python
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import shutil
import os



PDF_PATH = "Books_PDFs/"              # Directory of  PDF files
CHROMA_PATH = "Vector_Database/"      # Directory for saving the vector database


# OPEN AI API key in use
Open_ai = "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"



# Loading PDF files
def load_data():
    data_loader = PyPDFDirectoryLoader(PDF_PATH)
    loaded_data = data_loader.load()
    return loaded_data



# RECURSIVE CHARACTER TEXT SPLITTER 
def split_data(loaded_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 200,
        length_function = len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(loaded_data)
    print(f"Split {len(loaded_data)} data sets into {len(chunks)} chunks.")

    return chunks



# Saving the chunks to the vector database
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    vector_store = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=Open_ai), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")



PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def prompt():
    # Ask the question.
    query_text = input("Ask any question from the testing_data:  \n \t ")

    # Prepare the vector Database to answer the question.
    embedding_function = OpenAIEmbeddings(openai_api_key=Open_ai)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Searching the Database.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # prompting the openai model
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)


    model = ChatOpenAI(openai_api_key=Open_ai)
    response_text = model.invoke(prompt)
    print(response_text)

def ask_again():
    again = input("Would you like to ask another question? (yes/no) \n \t")
    if again.lower() == "yes":
        prompt()
    else:
        print("Thanks for using our service. Goodbye!")

# main function that executes all other functions
def main():
    documents = load_data()
    chunks = split_data(documents)
    save_to_chroma(chunks)
    prompt()
    ask_again()        


# calling the main function    
if __name__ == "__main__":
    main()   