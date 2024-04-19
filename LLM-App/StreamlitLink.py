#!/usr/bin/env python
# coding: utf-8

# #### Link To Access Streamlit App

# In[ ]:


import json
port='8501'
with open('/opt/ml/metadata/resource-metadata.json') as openfile:
        data = json.load(openfile)
domain_id = data['DomainId']
region=data['ResourceArn'].split(':')[3]
print(f'https://{domain_id}.studio.{region}.sagemaker.aws/jupyter/default/proxy/{port}/')

