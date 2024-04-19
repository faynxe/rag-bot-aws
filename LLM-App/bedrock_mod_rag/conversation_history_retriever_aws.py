import boto3
DYNAMODB  = boto3.resource('dynamodb')
def put_db(messages,table,user,session_id):
    """Store long term chat history in DynamoDB"""    
    chat_item = {
        "UserId": user, # user id
        "SessionId": session_id, # User session id
        "messages": [messages],  # 'messages' is a list of dictionaries
        "time":messages['time']
    }
    existing_item = DYNAMODB.Table(table).get_item(Key={"UserId": user, "SessionId":session_id})
    if "Item" in existing_item:
        existing_messages = existing_item["Item"]["messages"]
        chat_item["messages"] = existing_messages + [messages]
    response = DYNAMODB.Table(table).put_item(
        Item=chat_item
    )    
    
    
def get_chat_history_db(table,user,session_id,cutoff):
    current_chat=[]   
    # Retrieve past chat history from Dynamodb  
    chat_histories = DYNAMODB.Table(table).get_item(Key={"UserId": user, "SessionId":session_id})
    if "Item" in chat_histories:
        chat_hist=chat_histories['Item']['messages'][-cutoff:]            
        for d in chat_hist:            
            current_chat.append({'role': 'user', 'content': [{"type":"text","text":d['user']}]})
            current_chat.append({'role': 'assistant', 'content': d['assistant']})  
    return current_chat

def get_chat_historie_for_streamlit(table, user_id, session_id):
    """
    This function retrieves chat history stored in a dynamoDB table partitioned by a userID and sorted by a SessionID
    """
    chat_histories = DYNAMODB.Table(table).get_item(Key={"UserId": user_id, "SessionId":session_id})

# Constructing the desired list of dictionaries
    formatted_data = []
    if 'Item' in chat_histories:
        for entry in chat_histories['Item']['messages']:
            # assistant_attachment = entry.get('image', []) + entry.get('document', [])
            assistant_attachment = '\n\n'.join(entry.get('image', []) + entry.get('document', []))
            formatted_data.append({
                "role": "user",
                "content": entry["user"],
            })
            formatted_data.append({
                "role": "assistant",
                "content": entry["assistant"],
                "attachment": assistant_attachment
            })
    else:
        chat_histories=[]            
    return formatted_data,chat_histories

