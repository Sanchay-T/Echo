# import whisper
import os
print("Inside AI")
from dotenv import load_dotenv
load_dotenv()
# print(whisper.available_models())
# stt_model = whisper.load_model('small')
mykey = os.environ.get('OPENAI_API_KEY')
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from elevenlabs import generate, play, set_api_key , stream
import os
import datetime
import openai
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI, ChatAnthropic
import keyboard
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

from langchain import PromptTemplate
from langchain.agents import AgentType

import re





# load_dotenv("../.env")
# Set API keys
set_api_key(os.getenv("ELEVENLABS_API_KEY"))
set_api_key("dcb21c9f4a8176f2f19148f63cde21e4")
# openai_api_key = "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"
APP_KEY = os.getenv("DOLBY_APP_KEY")
APP_SECRET = os.getenv("DOLBY_APP_SECRET")
anthropic = os.getenv("ANTHROPIC_API_KEY")
# LANGCHAIN_TRACING_V2 = True
# LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
# LANGCHAIN_API_KEY = "pt-wilted-try-29"


memory = ConversationBufferMemory(memory_key="chat_history")



import speech_recognition as sr
r = sr.Recognizer()           


# Import the AssemblyAI module
import assemblyai as aai

# Your API token is already set here
aai.settings.api_key = "6c7f4d60028e4df9b889b93acb8ed698"
# 
# Create a transcriber object.

# transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/espn-bears.m4a%22")

transcriber = aai.Transcriber()

def stt_function(audio_path):
    audio_file= open(audio_path, "rb")

    transcript = transcriber.transcribe(audio_path)

    return transcript.text



def llm_reply(query):
    print("Inside LLM")
    print(query)
    response = llm_chain.run(query)
    print(response)
    return response




# %%
# Template_1

#
# system_message_prompt = SystemMessagePromptTemplate.from_template(template_1)
# human_template = "{query}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# chat_prompt = ChatPromptTemplate.from_messages(
#     [system_message_prompt, human_message_prompt]
# )
# # %%
# Assigning model and tools
chat = ChatAnthropic()
llm = ChatOpenAI(model="gpt-4", temperature=0)

zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
zapier_agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

tools_list = [f"{tool.name}: {tool.description}" for tool in toolkit.get_tools()]
final_string1 = "\n".join(tools_list)

baseprompt = """

Assistant is a large language model.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant is aware that human input is being transcribed from audio and as such there may be some errors in the transcription. It will attempt to account for some words being swapped with similar-sounding words or phrases. Assistant will also keep responses concise, because human attention spans are more limited over the audio channel since it takes time to listen to a response.


Assistant is going create the final proper output , you are personal chatbot who is going to have a conversation with the user and collect all the required information for the tools to work.
These are the tools i am using and the user input should contain all the required information for the tools to work.You  need to chat with the user until all the informtion for that specific tool is collected. 
The final answer should be generated when you have all the details ONLY , do not generate the final message until then , and your geneated final answer should be command to another person.

Slack: Send Direct Message: A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "send a slack message to the #general channel". Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. For example, if the params are ['Message_Text', 'Channel'], your instruction should be something like 'send a slack message to the #general channel with the text hello world'. Another example: if the params are ['Calendar', 'Search_Term'], your instruction should be something like 'find the meeting in my personal calendar at 3pm'. Do not make up params, they will be explicitly specified in the tool description. If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. If you get a none or null response, STOP EXECUTION, do not try to another tool!This tool specifically used for: Slack: Send Direct Message, and has params: ['To_Username', 'Message_Text']
Notion: Create Page: A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "send a slack message to the #general channel". Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. For example, if the params are ['Message_Text', 'Channel'], your instruction should be something like 'send a slack message to the #general channel with the text hello world'. Another example: if the params are ['Calendar', 'Search_Term'], your instruction should be something like 'find the meeting in my personal calendar at 3pm'. Do not make up params, they will be explicitly specified in the tool description. If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. If you get a none or null response, STOP EXECUTION, do not try to another tool!This tool specifically used for: Notion: Create Page, and has params: ['Parent_Page']
Zoom: Create Meeting: A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "send a slack message to the #general channel". Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. For example, if the params are ['Message_Text', 'Channel'], your instruction should be something like 'send a slack message to the #general channel with the text hello world'. Another example: if the params are ['Calendar', 'Search_Term'], your instruction should be something like 'find the meeting in my personal calendar at 3pm'. Do not make up params, they will be explicitly specified in the tool description. If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. If you get a none or null response, STOP EXECUTION, do not try to another tool!This tool specifically used for: Zoom: Create Meeting, and has params: ['Meeting_Type', 'When', 'Topic', 'Duration__in_minutes']
Google Meet: Schedule a Meeting: A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "send a slack message to the #general channel". Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. For example, if the params are ['Message_Text', 'Channel'], your instruction should be something like 'send a slack message to the #general channel with the text hello world'. Another example: if the params are ['Calendar', 'Search_Term'], your instruction should be something like 'find the meeting in my personal calendar at 3pm'. Do not make up params, they will be explicitly specified in the tool description. If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. If you get a none or null response, STOP EXECUTION, do not try to another tool!This tool specifically used for: Google Meet: Schedule a Meeting, and has params: ['Calendar', 'Start_Date___Time', 'End_Date___Time']
Gmail: Find Email: A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "send a slack message to the #general channel". Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. For example, if the params are ['Message_Text', 'Channel'], your instruction should be something like 'send a slack message to the #general channel with the text hello world'. Another example: if the params are ['Calendar', 'Search_Term'], your instruction should be something like 'find the meeting in my personal calendar at 3pm'. Do not make up params, they will be explicitly specified in the tool description. If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. If you get a none or null response, STOP EXECUTION, do not try to another tool!This tool specifically used for: Gmail: Find Email, and has params: ['Search_String']
Gmail: Reply to Email: A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "send a slack message to the #general channel". Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. For example, if the params are ['Message_Text', 'Channel'], your instruction should be something like 'send a slack message to the #general channel with the text hello world'. Another example: if the params are ['Calendar', 'Search_Term'], your instruction should be something like 'find the meeting in my personal calendar at 3pm'. Do not make up params, they will be explicitly specified in the tool description. If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. If you get a none or null response, STOP EXECUTION, do not try to another tool!This tool specifically used for: Gmail: Reply to Email, and has params: ['Thread', 'To', 'Body']
Gmail: Send Email: A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, for example "get the latest email from my bank" or "send a slack message to the #general channel". Each tool will have params associated with it that are specified as a list. You MUST take into account the params when creating the instruction. For example, if the params are ['Message_Text', 'Channel'], your instruction should be something like 'send a slack message to the #general channel with the text hello world'. Another example: if the params are ['Calendar', 'Search_Term'], your instruction should be something like 'find the meeting in my personal calendar at 3pm'. Do not make up params, they will be explicitly specified in the tool description. If you do not have enough information to fill in the params, just say 'not enough information provided in the instruction, missing <param>'. If you get a none or null response, STOP EXECUTION, do not try to another tool!This tool specifically used for: Gmail: Send Email, and has params: ['Subject', 'Cc', 'To', 'Body']

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Slack: Send Direct Message, Google Meet: Schedule a Meeting, Gmail: Find Email, Gmail: Send Email, Gmail: Send Email]
Action Input: Checking if all the required information is collected.If you don't have full information ask for it from the user.
Observation: Yes all the information needed for the tool is collected.
If the final condition is not met and you dont have all the information then print this:
Incomplete : [ Print here what you need to ask the user to provide the incomplete information ]
[ When condition is satisfied then dont print anything else just print this : 
Final Answer: the final answer to the original input question. ```Example : Send email to [email] with subject [subject] and body [body]```
]
[When the user asks something that is not being instructed for you and is something different that automating tasks with Zapier and tools then print this:
NORMAL ASSISTANT : [ And the answer to the question asked by the user ]  
]
When you have all the required information for the task , then send only the Final Answer as the output.Don't send any other thing as output from these [Action , Thought  , Question , Observation , Action Input]

Begin!
{chat_history}
Question: {question}


"""





prompt_template = PromptTemplate(
    input_variables=["chat_history","question"],
    template=baseprompt,
)

# prompt_template.format()


# chat = ChatOpenAI(temperature=0)
llm_chain = LLMChain(
    llm=chat,
    prompt=prompt_template,
    verbose=True,
    memory=memory
)


import base64

def play_generated_audio(text, voice="Bella", model="eleven_monolingual_v1"):
    audio = generate(text=text, voice=voice, model=model)
    
    bytes_array = list(audio)  # Convert audio bytes to a list of integers

    audio_bytes = bytearray(bytes_array) # Your audio bytes here

    # Convert byte array to base64 string
    base64_bytes = base64.b64encode(audio_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

my_streamit_prompt = ""



def llm_run_query(user_input):
    final_answer = None
    output = llm_chain.predict(question=user_input)
    print(type(output))
    print(4 * "-")
    print(output)

    inputs = re.search("Action Input: (.*)", output)
    # match_incomplete = re.search("Incomplete : (.*)", output)
    str = 'Incomplete: What should the subject and body of the email be?'
    regex = re.compile('Incomplete:\s*(.*)')
    match_ = regex.search(str)
    result = match_.group(1) if match_ else ''
    print("Incomplwt result ",result)  # 

    if match_:
        # action_input_text = match_incomplete.group(1)
        action_input_text = match_.group(1)
        # play_generated_audio(action_input_text)
        my_streamit_prompt = action_input_text
        return action_input_text


    match = re.search("Final Answer: (.*)", output)
    if match:
        final_answer = match.group(1)
        print("Final Answer ", final_answer)
        assistant_message = zapier_agent.run(final_answer)
        # play_generated_audio(assistant_message)

        return assistant_message

        # Ask the user if they have more questions.
        # more_questions = input("Do you have any more questions? (yes/no): ")
        # if more_questions.lower() != 'yes':
        #     pass

    else:
        return "No final answer found, please ask another question or provide more information."
    


from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = zapier_agent.run(prompt, callbacks=[st_callback])
        st.write(response)
