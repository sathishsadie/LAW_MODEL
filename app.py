import streamlit as st
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import re  # ✅ For cleaning model response

st.title("📜 Law-Based Answering System")

VECTOR_DB_FOLDER = "bns_vector_db_up_oct11"

# Load JSON file
def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

# Clean text
def clean_and_process_text(text_list):
    cleaned_text = " ".join(text_list)
    cleaned_text = " ".join(cleaned_text.split())  
    return cleaned_text

# Load data
bns_dic = load_json("bns.json")
bns_to_ipc = load_json("bns_to_ipc.json")

# Load embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Retrieve relevant legal sections
def retrieve_relevant_docs(vector_store, query, k=5):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': k})
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    if not docs_with_scores:
        return None
    return [doc for doc, score in docs_with_scores]

ipc_sections = []

def build_rag_chain():
    prompt = '''
        ### 🔹__CONTEXT__ - AI Legal Assistant - Bharatiya Nyaya Sanhita (BNS) Expert. 
            - You are a compassionate and expert legal AI specializing in the Bharatiya Nyaya Sanhita (BNS) 2023. 
            - Your responsibility is to *empathetically analyze legal situations* shared by a victim, a concerned friend, a police officer, or a legal professional, and provide *precise mapping to relevant BNS sections, along with the **appropriate punishments for the offender*. 


        ### 🔹 __ASK__ - Understanding the Case with Sensitivity:  
            - You are a legal expert who communicates empathetically.
            - Carefully identify and differentiate the situations or offenses faced by the victim, regardless of who describes the case.  
            - Clearly distinguish between the victim (the sufferer) and the offender (the person responsible for the offense).  
            - Recognize the emotional, psychological, and physical impact of each situation on the victim.  
            - Use the retrieved legal context(BNS sections) and interpret them in a way that aligns with the circumstances faced by the victim and ensure that the offender's punishment will be there based on legal provinsion.  
            - For each situation, map it to the most suitable BNS Section(s).  
            - Emphasize justice for the victim and ensure that the offender s punishment is detailed ased on the legal provision.  


        ### 🔹 __RESPONSE__ - Structured Legal Mapping:
            #### ONLY PROVIDE THE MOST RELEVANT LEGAL SECTION FOR EACH SITUATION and MAP THE LEGAL SECTIONS PUNISHMENT TO EACH SITUATION
                ✅ Mapped BNS Section(s) for Each Situation Faced by the Victim and Punishment (As per BNS):  
                    - BNS Section X(Y) and how is relavant to that situation . 
                    - For each BNS Section X(Y), provide the applicable legal punishment to the offender.  
                    - Include imprisonment term, fine, or both as per law. 

        ### 🔹__INPUT__ :  
            #### Case Description:  
                {question}  
            #### Retrieved Legal data (Relevant BNS Sections):  
                {legal_context}  

        ### 🔹 NOTE:
        - DO NOT include any internal thinking, steps like "<think>" or personal reflections such as "I'm trying to figure out..."
    '''
    prompt_template = ChatPromptTemplate.from_template(prompt)

    model = ChatOllama(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0.1,
        stream=True
    )

    return (
        {"legal_context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

question = st.text_input("Enter your legal situation:", placeholder="Describe your legal issue...")
retrieved = []

if st.button("Get Legal Advice") and question:
    legal_context = ""
    vector_db_path = Path(VECTOR_DB_FOLDER) 

    if vector_db_path.exists():
        vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)
        retrieved_docs = retrieve_relevant_docs(vector_store, question)

        if retrieved_docs:
            sections = list(set(doc.metadata.get('section') for doc in retrieved_docs))[:4]
            if len(sections) < 2:
                sections.append(str(int(sections[-1]) + 1))
                sections.append(str(int(sections[-1]) - 1))
            for section in sections:
                retrieved.append(f"### BNS(Bharatiya Nyaya Sanhita) Section: {section}\n" + "\nBNS section " + clean_and_process_text(bns_dic[section]))
                ipc_sections.append(f"BNS section {section}" + " ------> Corresponding IPC Sections " + clean_and_process_text(bns_to_ipc[section]))
            legal_context = "\n\n\n".join(retrieved)
        print(legal_context)
    # Generate AI response
    if legal_context:
        st.subheader("🎯 Comprehensive Legal Guidance")
        response_placeholder = st.empty()
        response = ""

        rag_chain = build_rag_chain()
        for chunk in rag_chain.stream({
            "legal_context": legal_context,
            "question": question
        }):
            response += chunk
            cleaned_response = re.sub(r"<think>.*?(?=\n|$)", "", response, flags=re.DOTALL)
            cleaned_response = cleaned_response.replace("Okay, so I'm trying to figure out what to do here", "").strip()


            response_placeholder.markdown(cleaned_response.replace("$", "\\$"))
    else:
        st.warning("No relevant legal documents found for your query.")

with st.expander("📜 Corresponding IPC Sections (Mapped from BNS)"):
    st.markdown("\n\n".join(ipc_sections))

st.subheader("🗂 Relevant Legal Sections (From BNS Document)")
with st.expander("Click to view Legal details"):
    st.markdown("\n\n".join(retrieved))