import streamlit as st
from llm_func import load_model, get_response

def page():

    # llm = load_model()
    st.title("🤖 P.A.I. Code Assistant")
    st.markdown("""
    **Programmable AI Interface (P.A.I.)** An intelligent coding companion powered by `Qwen2.5-Coder-0.5B`. 
    
    * **Capabilities:** Debugging, Code Generation, and Logic Optimization.
    * **Framework:** Built with LangChain & Streamlit.
    ---
    """)

    with st.container(border = True):
        if "llm" not in st.session_state:
            st.session_state.llm = load_model()

        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # display the previous messages on every page st.rerun() called in background
        for role,text in st.session_state.messages:

            # langchaing says role as ai/human rather than streamlit's prefered assistant and human
            if role == "system":
                continue
            st_role = "assistant" if role == "ai" else "user"

            with st.chat_message(st_role):
                st.markdown(text)

        if prompt := st.chat_input("Enter the prompt you want for the code you want to Generate"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
        
            with st.chat_message("assistant"):
                response = f"P.A.I." + get_response(st.session_state.llm, st.session_state.messages, prompt)
                st.markdown(response)
            
            st.session_state.messages.append(("human",prompt))
            st.session_state.messages.append(("ai", response))  # add user and humna messages to continue conversation memory
        

        

if __name__ == "__main__":
    page()