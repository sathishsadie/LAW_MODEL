# import streamlit as st
# import pandas as pd
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # Streamlit UI Title
# st.title("📜 Law-Based Answering System")

# # Vector Database Folder
# VECTOR_DB_FOLDER = "vector_db(removednewline)"
# vector_db_files = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")]

# # Initialize Embedding Model
# embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# # Function to Retrieve Relevant Documents
# def retrieve_relevant_docs(vector_store, query, threshold=0.5, k=5):
#     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': k})
#     docs_with_scores = vector_store.similarity_search_with_score(query, k=k)

#     relevant_docs = [(doc, score) for doc, score in docs_with_scores if score >= threshold]
#     if not relevant_docs:
#         return None

#     relevant_docs.sort(key=lambda x: x[1], reverse=True)
#     return relevant_docs

# # Function to Build RAG Chain
# def build_rag_chain():
#     prompt = """
#     You are a legal assistant specializing in law and legal procedures. Analyze the user's legal issue 
#     and provide a response strictly based on the retrieved legal context. 
#     **User Query:** {question}
#     **Source Document:** {document_name}
    
#     ## Compare the user **query** with the provided **context** and show only the **most relevant information**.
#     Provide a **step-by-step explanation** and **practical legal guidance**.
#     """
    
#     prompt_template = ChatPromptTemplate.from_template(prompt)
#     model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    
#     return (
#         {"context": RunnablePassthrough(), 
#          "question": RunnablePassthrough(),
#          "document_name": RunnablePassthrough()}
#         | prompt_template
#         | model
#         | StrOutputParser()
#     )

# # Streamlit Interface
# question = st.text_input("Enter your legal situation:", placeholder="Describe your legal issue...")

# if st.button("Get Legal Advice") and question:
#     results = []
#     for vector_db in vector_db_files:
#         vector_db_path = Path(VECTOR_DB_FOLDER) / f"{vector_db}.faiss"
#         if not vector_db_path.exists():
#             continue

#         vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)

#         with st.spinner(f"Analyzing {vector_db}..."):
#             retrieved_docs = retrieve_relevant_docs(vector_store, question)
#             if not retrieved_docs:
#                 continue

#             # Format extracted document content with section name first
#             formatted_contexts = []
#             for doc, _ in retrieved_docs:
#                 section_name = doc.metadata.get("section", "Unknown Section")
#                 formatted_contexts.append(f"### Section: {section_name}\n{doc.page_content}")

#             formatted_text = "\n\n".join(formatted_contexts)
#             doc_names = ", ".join({doc.metadata.get("source", vector_db) for doc, _ in retrieved_docs})

#             # Display Retrieved Context to User
#             st.subheader(f"📄 Retrieved Context from {doc_names}")
#             with st.expander("Click to view retrieved legal documents"):
#                 st.markdown(formatted_text)  # Show formatted retrieved text

#             # Generate response
#             rag_chain = build_rag_chain()
#             response = rag_chain.invoke({
#                 "context": formatted_text,
#                 "question": question,
#                 "document_name": doc_names
#             })
#             results.append((doc_names, response, formatted_text))

#     # Display Most Relevant Legal Analysis
#     if results:
#         st.subheader("🎯 Most Relevant Legal Analysis")
#         top_doc, top_response, top_context = results[0]

#         st.markdown(f"**Source Document:** {top_doc}")
#         st.markdown("---")
#         st.markdown(top_response)

#         # Show additional results
#         if len(results) > 1:
#             st.subheader("📚 Additional Relevant Sources")
#             for doc_name, response, context in results[1:]:
#                 with st.expander(f"View analysis from {doc_name}"):
#                     st.markdown(response)
#                     st.markdown("#### 📖 Retrieved Context")
#                     st.markdown(context)  # Show context under each result
#     else:
#         st.warning("No relevant legal documents found for your query.")



# import streamlit as st
# import pandas as pd
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # Streamlit UI Title
# st.title("📜 Law-Based Answering System")

# # Vector Database Folder
# VECTOR_DB_FOLDER = "vector_db(removednewline)"
# vector_db_files = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")]

# # Initialize Embedding Model
# embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# # Function to Retrieve Relevant Documents and Their Neighbors
# def retrieve_relevant_docs_with_neighbors(vector_store, query, k=5):
#     """
#     Retrieve relevant documents from FAISS along with the previous and next documents.
#     """
#     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': k})
#     docs_with_scores = vector_store.similarity_search_with_score(query, k=k)

#     if not docs_with_scores:
#         return None

#     # Sort documents based on score (highest first)
#     docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
#     retrieved_docs = []
#     seen_indices = set()  # Track already added documents

#     for doc, score in docs_with_scores:
#         section = doc.metadata.get("section", "Unknown Section")
#         index = doc.metadata.get("index", -1)

#         # Fetch the main document
#         retrieved_docs.append(doc)
#         seen_indices.add((section, index))

#         # Fetch the previous document (if exists)
#         prev_index = index - 1
#         if prev_index >= 0 and (section, prev_index) not in seen_indices:
#             prev_doc = vector_store.docstore._dict.get(f"{section}_{prev_index}")
#             if prev_doc:
#                 retrieved_docs.append(prev_doc)
#                 seen_indices.add((section, prev_index))

#         # Fetch the next document (if exists)
#         next_index = index + 1
#         next_doc = vector_store.docstore._dict.get(f"{section}_{next_index}")
#         if next_doc:
#             retrieved_docs.append(next_doc)
#             seen_indices.add((section, next_index))

#     return retrieved_docs

# # Function to Build RAG Chain
# def build_rag_chain():
#     prompt = """
#     You are a legal assistant specializing in law and legal procedures. Analyze the user's legal issue 
#     and provide a response strictly based on the retrieved legal context. 
#     **User Query:** {question}
#     **Source Document:** {document_name}
    
#     ## Compare the user **query** with the provided **context** and show only the **most relevant information**.
#     Provide a **step-by-step explanation** and **practical legal guidance**.
#     """
    
#     prompt_template = ChatPromptTemplate.from_template(prompt)
#     model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    
#     return (
#         {"context": RunnablePassthrough(), 
#          "question": RunnablePassthrough(),
#          "document_name": RunnablePassthrough()}
#         | prompt_template
#         | model
#         | StrOutputParser()
#     )

# # Streamlit Interface
# question = st.text_input("Enter your legal situation:", placeholder="Describe your legal issue...")

# if st.button("Get Legal Advice") and question:
#     results = []
#     for vector_db in vector_db_files:
#         vector_db_path = Path(VECTOR_DB_FOLDER) / f"{vector_db}.faiss"
#         if not vector_db_path.exists():
#             continue

#         vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)

#         with st.spinner(f"Analyzing {vector_db}..."):
#             retrieved_docs = retrieve_relevant_docs_with_neighbors(vector_store, question)
#             if not retrieved_docs:
#                 continue

#             # Format extracted document content with section name first
#             formatted_contexts = []
#             for doc in retrieved_docs:
#                 section_name = doc.metadata.get("section", "Unknown Section")
#                 formatted_contexts.append(f"### Section: {section_name}\n{doc.page_content}")

#             formatted_text = "\n\n".join(formatted_contexts)
#             doc_names = ", ".join({doc.metadata.get("source", vector_db) for doc in retrieved_docs})

#             # Display Retrieved Context to User
#             st.subheader(f"📄 Retrieved Context from {doc_names}")
#             with st.expander("Click to view retrieved legal documents"):
#                 st.markdown(formatted_text)  # Show formatted retrieved text

#             # Generate response
#             rag_chain = build_rag_chain()
#             response = rag_chain.invoke({
#                 "context": formatted_text,
#                 "question": question,
#                 "document_name": doc_names
#             })
#             results.append((doc_names, response, formatted_text))

#     # Display Most Relevant Legal Analysis
#     if results:
#         st.subheader("🎯 Most Relevant Legal Analysis")
#         top_doc, top_response, top_context = results[0]

#         st.markdown(f"**Source Document:** {top_doc}")
#         st.markdown("---")
#         st.markdown(top_response)

#         # Show additional results
#         if len(results) > 1:
#             st.subheader("📚 Additional Relevant Sources")
#             for doc_name, response, context in results[1:]:
#                 with st.expander(f"View analysis from {doc_name}"):
#                     st.markdown(response)
#                     st.markdown("#### 📖 Retrieved Context")
#                     st.markdown(context)  # Show context under each result
#     else:
#         st.warning("No relevant legal documents found for your query.")



import streamlit as st
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Streamlit UI Title
st.title("📜 Law-Based Answering System")

# Vector Database Folder
VECTOR_DB_FOLDER = "bnss_vector_db"
vector_db_files = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")]

# Initialize Embedding Model
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Function to Retrieve Relevant Documents Along with Their Neighbors
def retrieve_relevant_docs_with_neighbors(vector_store, query, k=5):
    """
    Retrieve relevant documents from FAISS along with the previous and next documents for each.
    """
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': k})
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)

    if not docs_with_scores:
        return None

    # Sort documents based on score (highest first)
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    retrieved_docs = []
    seen_indices = set()  # Track already added documents

    for doc, score in docs_with_scores:
        section = doc.metadata.get("section", "Unknown Section")
        index = doc.metadata.get("index", -1)

        # Fetch the main document
        if (section, index) not in seen_indices:
            retrieved_docs.append(doc)
            seen_indices.add((section, index))

        # Fetch the previous document (if exists)
        prev_index = index - 1
        prev_doc_key = f"{section}_{prev_index}"
        if prev_index >= 0 and (section, prev_index) not in seen_indices:
            prev_doc = vector_store.docstore._dict.get(prev_doc_key)
            if prev_doc:
                retrieved_docs.append(prev_doc)
                seen_indices.add((section, prev_index))

        # Fetch the next document (if exists)
        next_index = index + 1
        next_doc_key = f"{section}_{next_index}"
        next_doc = vector_store.docstore._dict.get(next_doc_key)
        if next_doc:
            retrieved_docs.append(next_doc)
            seen_indices.add((section, next_index))

    return retrieved_docs

# Function to Build RAG Chain
def build_rag_chain():
    prompt = """
    You are a legal assistant specializing in law and legal procedures. Analyze the user's legal issue 
    and provide a response strictly based on the retrieved legal context. 
    **User Query:** {question}
    **Source Document:** {document_name}
    
    ## Compare the user **query** with the provided **context** and show only the **most relevant information**.
    Provide a **step-by-step explanation** and **practical legal guidance**.
    """
    
    prompt_template = ChatPromptTemplate.from_template(prompt)
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    
    return (
        {"context": RunnablePassthrough(), 
         "question": RunnablePassthrough(),
         "document_name": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

# Streamlit Interface
question = st.text_input("Enter your legal situation:", placeholder="Describe your legal issue...")

if st.button("Get Legal Advice") and question:
    results = []
    for vector_db in vector_db_files:
        vector_db_path = Path(VECTOR_DB_FOLDER) / f"{vector_db}.faiss"
        if not vector_db_path.exists():
            continue

        vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)

        with st.spinner(f"Analyzing {vector_db}..."):
            retrieved_docs = retrieve_relevant_docs_with_neighbors(vector_store, question)
            if not retrieved_docs:
                continue

            # Format extracted document content with section name first
            formatted_contexts = []
            for doc in retrieved_docs:
                section_name = doc.metadata.get("section", "Unknown Section")
                formatted_contexts.append(f"### Section: {section_name}\n{doc.page_content}")

            formatted_text = "\n\n".join(formatted_contexts)
            doc_names = ", ".join({doc.metadata.get("source", vector_db) for doc in retrieved_docs})

            # Display Retrieved Context to User
            st.subheader(f"📄 Retrieved Context from {doc_names}")
            with st.expander("Click to view retrieved legal documents"):
                st.markdown(formatted_text)  # Show formatted retrieved text

            # Generate response
            rag_chain = build_rag_chain()
            response = rag_chain.invoke({
                "context": formatted_text,
                "question": question,
                "document_name": doc_names
            })
            results.append((doc_names, response, formatted_text))

    # Display Most Relevant Legal Analysis
    if results:
        st.subheader("🎯 Most Relevant Legal Analysis")
        top_doc, top_response, top_context = results[0]

        st.markdown(f"**Source Document:** {top_doc}")
        st.markdown("---")
        st.markdown(top_response)

        # Show additional results
        if len(results) > 1:
            st.subheader("📚 Additional Relevant Sources")
            for doc_name, response, context in results[1:]:
                with st.expander(f"View analysis from {doc_name}"):
                    st.markdown(response)
                    st.markdown("#### 📖 Retrieved Context")
                    st.markdown(context)  # Show context under each result
    else:
        st.warning("No relevant legal documents found for your query.")
