import webbrowser

import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if ".pdf" in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif ".docx" in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif ".pptx" in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=100, length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


# 사이드 바의 LLM 옵션 받기 위해 수정
def get_conversation_chain(vetorestore, openai_api_key, llm_option_1, llm_option_2):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=llm_option_1,
        max_tokens=llm_option_2,
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type="mmr", vervose=True),
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True,
    )
    return conversation_chain


def main():
    st.set_page_config(page_title="~AI 챗봇", page_icon=":books:")

    st.title("_개인 데이터 :red[QA 채팅]_ :books:")

    st.divider()

    st.subheader("안녕하세요?  ~ 상담 AI 챗봇입니다 ... ")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # 사이드 바, 탭 추가
    with st.sidebar:
        sidebar_tab1, sidebar_tab2 = st.tabs(["기본 탭", "추가 탭"])

        with sidebar_tab1:
            uploaded_files = st.file_uploader(
                "파일을 업로드하세요.", type=["pdf", "docx"], accept_multiple_files=True
            )

            st.divider()

            openai_api_key = st.text_input(
                "OpenAI API Key를 입력하세요", key="chatbot_api_key", type="password"
            )
            process = st.button("OpenAI API Key 적용")

            st.divider()

            # llm 옵션 추가
            llm_option1 = st.slider(
                "LLM 옵션1을 설정하세요",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
            )

            st.write("위 옵션 값이 높을 수록 다양한 답변을 생성합니다.")
            print("option 1: ", llm_option1)

            # st.divider()

            llm_option2 = st.select_slider(
                "LLM 옵션2를 설정하세요", options=[128, 256, 512, 1024, 2048]
            )
            st.write("위 옵션 값에 따라 최대 생성 문자 수가 결정됩니다.")
            print("option 2: ", llm_option2)

            st.divider()

            st.markdown("#### API Key를 발급받는 방법")

            # API Key 발급 링크로 연결
            if st.button("OpenAI API Key"):
                url = "https://m.blog.naver.com/mynameistk/223062993136"
                webbrowser.open_new_tab(url)

            if st.button("Google Search API Key"):
                url = "https://guide.bati.ai/service/api/googleapi"
                webbrowser.open_new_tab(url)

            if st.button("Hugging Face API Key"):
                url = "https://yunwoong.tistory.com/225"
                webbrowser.open_new_tab(url)

        with sidebar_tab2:
            st.header("추가 탭 1")
            st.markdown("#### 아직 기능을 구현하는 중입니다")

    if process:
        if not openai_api_key:
            st.info(
                "계속하려면 OpenAI API 키를 추가하세요."
            )  # Please add your OpenAI API key to continue.
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(
            vetorestore, openai_api_key
        )
        st.session_state.processComplete = True

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # if llm_option1_slider:
    #     pass

    # if llm_option2_slider:
    #     pass

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("생각하는 중입니다..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result["chat_history"]
                response = result["answer"]
                source_documents = result["source_documents"]

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(
                        source_documents[0].metadata["source"],
                        help=source_documents[0].page_content,
                    )
                    st.markdown(
                        source_documents[1].metadata["source"],
                        help=source_documents[1].page_content,
                    )
                    st.markdown(
                        source_documents[2].metadata["source"],
                        help=source_documents[2].page_content,
                    )
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
