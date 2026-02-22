# # run the open-source models
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
# import torch

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


# model = pipeline(task = "summarization", model = "facebook/bart-large-cnn")
# response = model("text to sumamrize")
# print(response)

model = pipeline(
    task = "text-generation",
    model = "mistralai/Mistral-7B-Instruct-v0.1", 
    device = 0, 
    max_length = 256, 
    truncation = True,
    # model_kwargs={
    #     "pad_token_id": 2,       
    #     "max_length": 256          
    # }
)

llm = HuggingFacePipeline(pipeline = model)

template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old to understand")

# langchain '|' chain use- take the output that you get from passing the template and pass it to the input of the next llm
chain = template | llm
topic = input("Topic:")
age = input("Age:")

response = chain.invoke({"topic":topic, "age":age})
print(response)
