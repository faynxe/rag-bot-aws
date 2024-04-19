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
import json
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import time
import random
import logging
from botocore.exceptions import ClientError
from bedrock_mod_rag import rag_bot_utilities_aws,bedrock_llm_caller_aws,conversation_history_retriever_aws,pre_retrieval_tech_aws

APP_MD    = json.load(open('application_metadata_complete.json', 'r'))
MODELS_LLM = {d['name']: d['endpoint'] for d in APP_MD['models-llm']}
MODELS_EMB = {d['name']: d['endpoint'] for d in APP_MD['models-emb']}
REGION    = APP_MD['region']
BUCKET    = APP_MD['Kendra']['bucket']
PREFIX    = APP_MD['Kendra']['prefix']
OS_ENDPOINT  =  APP_MD['opensearch']['domain_endpoint']
KENDRA_ID = APP_MD['Kendra']['index']
KENDRA_ROLE=APP_MD['Kendra']['role']
PARENT_TEMPLATE_PATH="prompt_template"
KENDRA_S3_DATA_SOURCE_NAME=APP_MD['Kendra']['s3_data_source_name']
# Read credentials
with open('config.json') as f:
    config_file = json.load(f)
DYNAMODB_TABLE=config_file["DynamodbTable"]
DYNAMODB_USER=config_file["UserId"]



DYNAMODB      = boto3.resource('dynamodb')
S3            = boto3.client('s3', region_name=REGION)
TEXTRACT      = boto3.client('textract', region_name=REGION)
KENDRA        = boto3.client('kendra', region_name=REGION)
SAGEMAKER     = boto3.client('sagemaker-runtime', region_name=REGION)
BEDROCK = boto3.client(service_name='bedrock-runtime',region_name='us-east-1') 
KENDRA_INDEX="b3b83a9c-0de7-4791-b8b8-28caf96f6161"
# Vector dimension mappings of each embedding model
EMB_MODEL_DICT={"titan":1536,
                "minilmv2":384,
                "bgelarge":1024,
                "gtelarge":1024,
                "e5largev2":1024,
                "e5largemultilingual":1024,
               "gptj6b":4096,
                "cohere":1024}

# Creating unique domain names for each embedding model using the domain name prefix set in the config json file
# and a corresponding suffix of the embedding model name
EMB_MODEL_DOMAIN_NAME={"titan":f"{APP_MD['opensearch']['domain_name']}_titan",
                "minilmv2":f"{APP_MD['opensearch']['domain_name']}_minilm",
                "bgelarge":f"{APP_MD['opensearch']['domain_name']}_bgelarge",
                "gtelarge":f"{APP_MD['opensearch']['domain_name']}_gtelarge",
                "e5largev2":f"{APP_MD['opensearch']['domain_name']}_e5large",
                "e5largemultilingual":f"{APP_MD['opensearch']['domain_name']}_e5largeml",
               "gptj6b":f"{APP_MD['opensearch']['domain_name']}_gptj6b",
                       "cohere":f"{APP_MD['opensearch']['domain_name']}_cohere"}

# Session state keys
if 'user_sess' not in st.session_state:
    st.session_state['user_sess'] =f"{DYNAMODB_USER}-{str(time.time()).split('.')[0]}"
if 'generate' not in st.session_state:
    st.session_state['generate'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
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
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0   
if 'bytes' not in st.session_state:
    st.session_state['bytes'] = None
if 'rtv' not in st.session_state:
    st.session_state['rtv'] = ''
if 'page_summ' not in st.session_state:
    st.session_state['page_summ'] = ''
if 'action_name' not in st.session_state:
    st.session_state['action_name'] = ""
if 'chat_memory' not in st.session_state:
    st.session_state['chat_memory']=""
if 'input_token' not in st.session_state:
    st.session_state['input_token']=0
if 'output_token' not in st.session_state:
    st.session_state['output_token']=0


def load_document(file_bytes, doc_name, params):
    s3_path=f"file_store/{doc_name}"
    file_bytes=file_bytes.read()
    S3.put_object(Body=file_bytes,Bucket= BUCKET, Key=s3_path)       
    time.sleep(1)    
    s3_uri=f"s3://{BUCKET}/{s3_path}"
    with io.BytesIO(file_bytes) as open_pdf_file:   
        doc = fitz.open(stream=open_pdf_file) 
    if doc.page_count>1:    
        text,config= rag_bot_utilities_aws.call_amazon_textractor_on_pdf(s3_uri, True, False)
    else:
        text,config= rag_bot_utilities_aws.call_amazon_textractor_on_pdf(s3_uri, True, False)
    parse_text,parse_table=rag_bot_utilities_aws.page_content_parser_aws(text,config)
    chunks, table_header_dict, chunk_header_mapping=rag_bot_utilities_aws.advanced_pdf_chunker_(params['chunk'], parse_text)
    rag_bot_utilities_aws.upload_pdf_chunk_to_s3(doc_name,chunk_header_mapping, BUCKET)
    params['pipeline_name']="normalization-pipe"
    rag_bot_utilities_aws.norm_pipeline(params['pipeline_name'],"min_max","arithmetic_mean")
    params['dimension']=EMB_MODEL_DICT[params['emb'].lower()]
    rag_bot_utilities_aws.opensearch_document_loader_aws(params,chunks,table_header_dict,doc_name)
    
    

    
def query_llm(params,prompt,handler):
    if "opensearch" in params['rag'].lower():
        params['pipeline_name']="normalization-pipe"
        response=rag_bot_utilities_aws.similarity_search(prompt, params)
        response,tables,doc_name=rag_bot_utilities_aws.content_extraction_os_(response, True, "passage", BUCKET,params)
    elif "kendra" in params['rag'].lower():
        response=rag_bot_utilities_aws.query_index(prompt,params)
        response,passage_list,doc_name=rag_bot_utilities_aws.content_extraction_os_(response, True, "passage", BUCKET,params)
        tables=""
    elif "knowledgebase" in params['rag'].lower():
        response=rag_bot_utilities_aws.query_kb_index(prompt,params)
        response,passage_list,doc_name=rag_bot_utilities_aws.content_extraction_os_(response, True, "passage", BUCKET,params)
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
    return answer, doc_name
        
        

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
        if params["memory"]:
            prompt=pre_retrieval_tech_aws.llm_decomposer(params,st.session_state.messages,prompt)
        st.session_state.messages.append({"role": "user", "content": prompt,})
        with st.chat_message("user"):
            st.markdown(prompt)      

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            answer, doc_name=query_llm(params,prompt,message_placeholder)
            message_placeholder.markdown(answer.replace("$","USD ").replace("%", " percent"))
            st.session_state.messages.append({"role": "assistant", "content": answer,"attachment":doc_name}) 
        st.rerun()


        
def app_sidebar():
    with st.sidebar:               
        description = """### AI tool powered by suite of AWS services"""
        st.write(description)
        st.text_input('Total Token Used', str(st.session_state['input_token']+st.session_state['output_token'])) 
        st.write('---')
        st.write('### User Preference')
        filepath=None
        
        llm_model_name = st.selectbox('Select LL Model', options=MODELS_LLM.keys())
        retriever = st.selectbox('Retriever', ["OpenSearch","Kendra","KnowledgeBase"])       
        K=st.slider('Top K Results', min_value=1., max_value=10., value=3., step=1.,key='kkendra')
        mem = st.checkbox('chat memory')
        if "OpenSearch" in retriever:
            embedding_model=st.selectbox('Embedding Model', MODELS_EMB.keys())
            knn=st.slider('Nearest Neighbour', min_value=1., max_value=100., value=3., step=1.)                
            engine=st.selectbox('KNN Library', ("nmslib", "lucene","faiss"), help="Underlying KNN algorithm implementation to use for powering the KNN search")
            if "nmslib" in engine:
                space_type=st.selectbox("KNN Space",("cosinesimil","l2","innerproduct",  "l1", "linf"))
            elif "lucene" in engine:
                space_type=st.selectbox("KNN Space",("l2", "cosinesimil"))
            elif "faiss" in engine:
                space_type=st.selectbox("KNN Space",("l2", "innerproduct"))
            m=st.slider('Neighbouring Points', min_value=16.0, max_value=124.0, value=72.0, step=1., help="Explored neighbors count")
            ef_search=st.slider('efSearch', min_value=10.0, max_value=2000.0, value=1000.0, step=10., help="Exploration Factor")
            ef_construction=st.slider('efConstruction', min_value=100.0, max_value=2000.0, value=1000.0, step=10., help="Explain Factor Construction")            
            chunk=st.slider('Word Chunk size', min_value=50, max_value=5000 if "titan" in embedding_model.lower() else 300, value=1000 if "titan" in embedding_model.lower() else 250, step=50,help="Word size to chunk documents into Vector DB") 
            st.session_state['domain']=EMB_MODEL_DOMAIN_NAME[embedding_model.lower()]

            params = {'endpoint-llm':MODELS_LLM[llm_model_name],'model_name':llm_model_name, "emb_model":MODELS_EMB[embedding_model], "rag":retriever,"K":K, "engine":engine, "m":m,
                     "ef_search":ef_search, "ef_construction":ef_construction, "chunk":chunk, "domain":st.session_state['domain'], "knn":knn,
                     'emb':embedding_model,"space_type":space_type,"memory":mem }   

        else:

            params = {'endpoint-llm':MODELS_LLM[llm_model_name],"K":K,'model_name':llm_model_name, "rag":retriever,"memory":mem }   

        file = st.file_uploader('Upload a PDF file', type=['pdf']) 
        if file is not None:
            file_name=str(file.name)
            st.session_state['file_name']=file_name
            st.session_state.generated.append(1)
            domain=load_document(file, file_name,params) 

    return params, file

def main():
    params,f = app_sidebar()
    action_doc(params)

if __name__ == '__main__':
    main()
  
