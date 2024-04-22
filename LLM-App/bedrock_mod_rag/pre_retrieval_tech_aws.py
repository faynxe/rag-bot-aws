from . import bedrock_llm_caller_aws
import streamlit as st

def llm_decomposer(params,current_chat,question):
    """ This function determines the context of each new question looking at the conversation history...
        to send the appropiate question to the retriever.    
        Messages are stored in DynamoDb if a table is provided or in memory in the absence of a provided DynamoDb table
    """
    system_prompt=""
    formatted_conversation = ""
    for entry in current_chat:
        role = entry["role"]
        content = entry["content"]
        formatted_conversation += f"{role}: {content}\n"

    memory_template=f"""Here is the history of a conversation dialogue between a user and an assistant. The last conversation being the most recent:
<conversation_dialouge>
{formatted_conversation}
</conversation_dialouge>

Here is a new question from the user:
user: {question}

Your task is to determine if the question is a follow-up to the recent conversations:
- If it is, rephrase the question as an independent question while retaining the original intent.
- If it is not, respond with "no".

Remember, your role is not to answer the question! Do not provide any preamble in your answer, just provide the rephrased question or "no" given the instruction above.

Format your response as:
<response>
answer
</response>"""
    if current_chat:        
        answer=bedrock_llm_caller_aws._invoke_bedrock_with_retries(params, [], system_prompt, memory_template, "anthropic.claude-3-sonnet-20240229-v1:0", )
        
        idx1 = answer.index('<response>')
        idx2 = answer.index('</response>')
        question_2=answer[idx1 + len('<response>') + 1: idx2]
        if 'no' != question_2.strip():
            question=question_2
        # print(question)     
    return question
