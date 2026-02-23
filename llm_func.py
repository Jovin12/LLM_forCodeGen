from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_model():
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype = "auto",
        device_map = "auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, max_new_tokens = 512)
    llm = HuggingFacePipeline(pipeline = pipe)
    return llm



def get_response(llm, prompt):

    # THIS IS THE HUGGIN FACE PIPELINE PROMPTS
    # messages = [
    #     {"role":"system", "content":"You are Qwen, a helpful programming assitant, to assist with and help users with their programs and errors they may get with the programs"},
    #     {"role": "user", "content": prompt}
    # ]

    # LANGCHAIN prefered prompt inputs
    template = ChatPromptTemplate.from_messages(
        [
            ("system","You are Qwen, a helpful programming assitant, to assist with and help users with their programs and errors they may get with the programs"),
            ("human", "{prompt}")
        ]
    )

    # using langchain to chain together multiple steps
    # must be string parsed in order to get the output froma tensor to string
    llm_chain = template | llm | StrOutputParser()
    response = llm_chain.invoke({"prompt":prompt})
    print(response)
