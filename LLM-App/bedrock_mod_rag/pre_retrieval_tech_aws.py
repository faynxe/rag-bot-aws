from . import bedrock_llm_caller_aws
import streamlit as st
def llm_decomposer(params,current_chat,question):
    """ This function determines the context of each new question looking at the conversation history...
        to send the appropiate question to the retriever.    
        Messages are stored in DynamoDb if a table is provided or in memory, in the absence of a provided DynamoDb table
    """
    system_prompt=""
    memory_template = f"""Your responsibility is to assess whether the provided question relates to the preceding dialogue between the assistant and a user:
- If it is a follow-up but already a complete, independent question retaining the original intent, then return the same question.
- If it is a follow-up but incomplete or vague, reformulate the query as a standalone question while preserving its original meaning.
- If it is not a follow-up question at all, simply respond with "_no_".
Remember, your responsibility is not to answer the question but determine whether it should be reformulated or not!
user: {question}
Format your response as:
<response>
answer
</response>"""
    if current_chat:
        current_chat= [{k: v for k, v in d.items() if k != "attachment"} for d in current_chat]
        # st.write(current_chat)
        answer=bedrock_llm_caller_aws._invoke_bedrock_with_retries(params, current_chat, system_prompt, memory_template, "anthropic.claude-3-sonnet-20240229-v1:0", )
        idx1 = answer.index('<response>')
        idx2 = answer.index('</response>')
        question_2=answer[idx1 + len('<response>') + 1: idx2]
        if '_no_' not in question_2:
            question=question_2
        print(question)
    return question