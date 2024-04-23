import os
import time
import json
import boto3
import pandas as pd
import logging
import streamlit as st
from pathlib import Path
import time
from PIL import Image
from io import BytesIO
# from PyPDF2 import PdfWriter, PdfReader
import requests
from tqdm import tqdm
import yaml
import multiprocessing
import uuid
import fitz
import io
import re
import json
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import time
import random
import concurrent.futures
import logging
from botocore.exceptions import ClientError
from bedrock_mod_rag import rag_bot_utilities_aws,bedrock_llm_caller_aws,conversation_history_retriever_aws,pre_retrieval_tech_aws

APP_MD    = json.load(open('application_metadata_complete.json', 'r'))
MODELS_LLM = {d['name']: d['endpoint'] for d in APP_MD['models-llm']}
MODELS_EMB = {d['name']: d['endpoint'] for d in APP_MD['models-emb']}
REGION    = APP_MD['region']
KENDRA_BUCKET    = APP_MD['Kendra']['bucket']
OPENSEARCH_BUCKET    = APP_MD['opensearch']['bucket']
KB_BUCKET=APP_MD['KnowledgeBase']['bucket']

# Read credentials
with open('config.json') as f:
    config_file = json.load(f)
DYNAMODB_TABLE=config_file["DynamodbTable"]
DYNAMODB_USER=config_file["UserId"]
DYNAMODB      = boto3.resource('dynamodb')
S3            = boto3.client('s3', region_name=REGION)

# Session state keys
if 'user_sess' not in st.session_state:
    st.session_state['user_sess'] =f"{DYNAMODB_USER}-{str(time.time()).split('.')[0]}"
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'domain' not in st.session_state:
    st.session_state['domain'] = ""
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = ''
if 'token' not in st.session_state:
    st.session_state['token'] = 0
if 'text' not in st.session_state:
    st.session_state['text'] = ''
if 'summary' not in st.session_state:
    st.session_state['summary']=''
if 'message' not in st.session_state:
    st.session_state['message'] = []
if 'bytes' not in st.session_state:
    st.session_state['bytes'] = None
if 'input_token' not in st.session_state:
    st.session_state['input_token']=0
if 'output_token' not in st.session_state:
    st.session_state['output_token']=0


def query_llm(params,prompt,handler):
    if "opensearch" in params['rag'].lower():
        params['pipeline_name']="normalization-pipe"
        response=rag_bot_utilities_aws.similarity_search(prompt, params)
        retrieved_section=params["hierach"].lower() if params["hierach"] else "passage"
        response,tables,doc_name=rag_bot_utilities_aws.content_extraction_os_(response, True, retrieved_section, OPENSEARCH_BUCKET,params)
    elif "kendra" in params['rag'].lower():
        response=rag_bot_utilities_aws.query_index(prompt,params)
        response,passage_list,doc_name=rag_bot_utilities_aws.content_extraction_os_(response, True, "passage", KENDRA_BUCKET,params)
        tables=""
    elif "knowledgebase" in params['rag'].lower():
        response=rag_bot_utilities_aws.query_kb_index(prompt,params)
        response,passage_list,doc_name=rag_bot_utilities_aws.content_extraction_os_(response, True, "passage", KENDRA_BUCKET,params)
        tables=""

    # st.write(response)
    with open("prompt_template/rag/claude/prompt1.txt","r")as f:
        prompt_template=f.read()
    values = {
        "passages": response,  
        "tab":tables,
        "question":prompt
        }    
    query=prompt_template.format(**values)
    answer=bedrock_llm_caller_aws._invoke_bedrock_with_retries(params,[], '', query, params['endpoint-llm'], [], handler)
    relevant_passges=[]
    pattern = r'<source>(.*?)</source>'
    matches = re.findall(pattern, answer, re.DOTALL)
    if matches:        
        for i, match in enumerate(matches, start=1):
            relevant_passges.append(match)
        pattern = r'<source>.*?</source>'
        answer = re.sub(pattern, '', answer, flags=re.DOTALL)
        relevant_passges=[x.split("_")[0] for x in relevant_passges]
        
    chat_history={"user":prompt,
    "assistant":answer,   
    "document":doc_name,
    "modelID":params['endpoint-llm'],
    "time":str(time.time()),
    "input_token":round(st.session_state['input_token']) ,
    "output_token":round(st.session_state['output_token'])} 
    #store convsation memory and user other items in DynamoDB table
    # conversation_history_retriever_aws.put_db(chat_history,DYNAMODB_TABLE,DYNAMODB_USER,st.session_state['user_sess'])
    # use local memory for storage   
    return answer, set(doc_name), relevant_passges
        
        

def action_doc(params):   
    # conversation_history=conversation_history_retriever_aws.get_chat_history_db(DYNAMODB_TABLE,DYNAMODB_USER,st.session_state['user_sess'],10)
    # st.session_state.messages,params['chat_histories']=conversation_history_retriever_aws.get_chat_historie_for_streamlit(DYNAMODB_TABLE, DYNAMODB_USER, st.session_state['user_sess'])
    st.title('Ask Questions of your Document')
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):         
            st.markdown(message["content"].replace("$", "\$"),unsafe_allow_html=True )
            if message["role"]=="assistant":
                if "attachment" in message and message["attachment"]:
                    with st.expander(label="**attachments**"):
                        st.markdown(message["attachment"])

    if prompt := st.chat_input(""):  
        if "Query Rewritting" in params["advanced_technique"]:
            prompt=pre_retrieval_tech_aws.llm_decomposer(params,st.session_state.messages,prompt)
        st.session_state.messages.append({"role": "user", "content": prompt,})
        with st.chat_message("user"):
            st.markdown(prompt)      

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            answer, doc_name, relevant_passges=query_llm(params,prompt,message_placeholder)
            doc_name="\n\n".join(doc_name)
            if relevant_passges:
                doc_name=f"**Source Documents**:\n\n{doc_name}\n\n**Relevant Passages**:  {','.join(relevant_passges)}"
            message_placeholder.markdown(answer.replace("$","USD ").replace("%", " percent"))
            st.session_state.messages.append({"role": "assistant", "content": answer,"attachment":doc_name}) 
        st.rerun()


        
def app_sidebar():
    with st.sidebar:               
        description = """### AI tool powered by suite of AWS services"""
        st.write(description)
        st.text_input('Total Bedrock LLM Token Used', str(st.session_state['input_token']+st.session_state['output_token'])) 
        st.write('---')
        st.write('### User Preference')
        filepath=None
        
        llm_model_name = st.selectbox('Select LL Model', options=MODELS_LLM.keys())
        retriever = st.selectbox('Retriever', ["OpenSearch","Kendra","KnowledgeBase"])  
        advanced_rag=st.multiselect("Advanced RAG Techniques",["Query Rewritting","ReRanking","Small2Big"], default=None)
        K=st.slider('Top K Results', min_value=1., max_value=50., value=10., step=1.,key='kkendra')
     
        if "OpenSearch" in retriever:
            hierach=""
            if "Small2Big" in advanced_rag:
                hierach=st.selectbox("Hierach",["Header", "Title"])
            search_type=st.selectbox('Search Type', ["hybrid","semantic","lexical"])          
            embedding_model=st.selectbox('Embedding Model', MODELS_EMB.keys())                          
            engine=st.selectbox('ANN Library', ("nmslib", "lucene","faiss"), help="Underlying KNN algorithm implementation to use for powering the KNN search")
            if "nmslib" in engine:
                space_type=st.selectbox("ANN Distance Type",("cosinesimil","l2","innerproduct",  "l1", "linf"))
            elif "lucene" in engine:
                space_type=st.selectbox("ANN Distance Type",("l2", "cosinesimil"))
            elif "faiss" in engine:
                space_type=st.selectbox("ANN Distance Type",("l2", "innerproduct"))
            knn=st.slider('Nearest Neighbour', min_value=1., max_value=100., value=3., step=1.)  
            with st.expander("Indexing Parameters"):                                
                m=st.slider('Neighbouring Points', min_value=16.0, max_value=124.0, value=72.0, step=1., help="Explored neighbors count")
                ef_search=st.slider('efSearch', min_value=10.0, max_value=2000.0, value=1000.0, step=10., help="Exploration Factor")
                ef_construction=st.slider('efConstruction', min_value=100.0, max_value=2000.0, value=1000.0, step=10., help="Explain Factor Construction")            
                chunk=st.slider('Word Chunk size', min_value=50, max_value=5000 if "titan" in embedding_model.lower() else 300, value=250, step=50,help="Word size to chunk documents into Vector DB") 
            params = {'endpoint-llm':MODELS_LLM[llm_model_name],'model_name':llm_model_name, "emb_model":MODELS_EMB[embedding_model], "rag":retriever,"K":K, "engine":engine, "m":m,
                     "ef_search":ef_search, "ef_construction":ef_construction, "chunk":chunk, "knn":knn,
                     'emb':embedding_model,"space_type":space_type,"advanced_technique":advanced_rag ,"search_type":search_type,"hierach":hierach}  
        
        elif "KnowledgeBase" in retriever:
            search_type=st.selectbox('Search Type', ["hybrid","semantic"])     
            params = {'endpoint-llm':MODELS_LLM[llm_model_name],"K":K,'model_name':llm_model_name, "rag":retriever,"advanced_technique":advanced_rag, "search_type":search_type}  
        else:
            params = {'endpoint-llm':MODELS_LLM[llm_model_name],"K":K,'model_name':llm_model_name, "rag":retriever,"advanced_technique":advanced_rag } 
        

        file = st.file_uploader('Upload a PDF file', type=['pdf'],accept_multiple_files=True) 
        if file: 
            file_name=[x.name for x in file] 
            st.session_state['file_name']=file_name
            st.session_state.generated.append(1)
            if "opensearch" in params['rag'].lower():
                domain=rag_bot_utilities_aws.parallelize_load_document(file, file_name,params) 
            else:
                domain=rag_bot_utilities_aws.load_document(file, file_name,params) 

    return params, file

def main():
    params,f = app_sidebar()
    if "engine" in params and params["engine"] == "lucene" and params['emb'].lower()=="titan":
        st.error("Lucene does not support up to 1500 for vector field dimension, use a different embedding model")
        st.stop()
    action_doc(params)

if __name__ == '__main__':
    main()
  
