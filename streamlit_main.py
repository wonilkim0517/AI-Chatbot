import os
import re
import shutil
import webbrowser
import unicodedata

import numpy as np
import streamlit as st

from pykospacing import Spacing
from transformers import BertTokenizer

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI  # 버전에 맞춰 다르게 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer, util

# from utils.functions import (
#     get_pdf_files,
#     remove_underscore_after_pdf,
#     chg_itemname,
#     copy_files_to_another_folder,
#     correct_spacing,
#     complete_sentence,
#     length_based_filtering,
# ) -> streamlit은  'st.set_page_config'가 무조건 먼저 실행되어야해서 이러면 오류났음


# 디렉토리 경로로 파일을 가져오기
def get_pdf_files(directory):
    file_list = os.listdir(directory)
    pdf_files = [file for file in file_list if file.endswith(".pdf")]
    return pdf_files


# ?
def remove_underscore_after_pdf(pdf_files):
    modified_pdf_files = []

    for file in pdf_files:
        base_name = os.path.splitext(file)[0]
        base_name_without_underscore = base_name.split("_")[0]
        new_file_name = f" {base_name_without_underscore}"
        modified_pdf_files.append(new_file_name)

    return modified_pdf_files


# 한글 자모 깨짐 방지
def chg_itemname(itemnames):
    itemlists = list()
    for itemname in itemnames:
        itemlists.append(unicodedata.normalize("NFC", itemname))
    return itemlists


# 가져온 파일을 DOC 폴더로 넣기
def copy_files_to_another_folder(
    source_folder, target_folder, files_to_copy
):  # 대상 폴더가 없으면 생성 / 폴더가 있으면 내용 비우기
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
        print('create "docs" folder')
    else:
        for f in os.listdir(target_folder):
            os.remove(os.path.join(target_folder, f))
        print("clear folder")

    for file_to_copy in files_to_copy:
        source_path = os.path.join(source_folder, file_to_copy)
        target_path = os.path.join(target_folder, file_to_copy)

        try:
            shutil.copy2(source_path, target_path)
            print(f"File '{file_to_copy}' copied successfully.")
        except FileNotFoundError:
            print(f"File '{file_to_copy}' not found in the source folder.")


# BERT 토크나이저를 이용한 한글 토큰 추출 함수
def extract_korean_tokens(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokens = tokenizer.tokenize(text)
    korean_tokens = [token for token in tokens if re.match("^[가-힣]+$", token)]
    return korean_tokens


# 띄어쓰기 오류 교정 함수
def correct_spacing(text):
    spacing = Spacing()
    corrected_text = spacing(text)
    return corrected_text


# 불완전 문장 처리 함수
def complete_sentence(text):
    # 문장의 끝에 마침표가 없으면 추가
    print("text: ", text)
    if text[-1] not in [".", "?", "!"]:
        text += "."
    return text


# 길이 기반 필터링 함수
def length_based_filtering(text, min_length=10):
    if len(text) > min_length:
        return text
    else:
        return ""


# BERT 임베딩을 사용하여 전처리 및 필터링을 수행하는 함수
def preprocess_with_bert_embedding(text):
    # 한글 토큰 추출
    korean_tokens = extract_korean_tokens(text)

    # 띄어쓰기 오류 교정
    corrected_text = correct_spacing(" ".join(korean_tokens))

    # 불완전 문장 처리
    completed_text = complete_sentence(corrected_text)

    # 길이 기반 필터링
    filtered_text = length_based_filtering(completed_text)

    return filtered_text


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


def main(origin_path, copy_path, top_k):
    st.set_page_config(page_title="AI 챗봇")

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
            upload_option = st.radio(
                "파일을 업로드할 방법 선택하세요", ("파일 선택", "경로 입력")
            )

            if upload_option == "파일 선택":
                uploaded_files = st.file_uploader(
                    "파일을 업로드하세요.",
                    type=["pdf", "docx", "pptx"],
                    accept_multiple_files=True,
                )
            elif upload_option == "경로 입력":
                directory_path = st.text_input("디렉토리 경로를 입력하세요.")
                if os.path.isdir(directory_path):
                    uploaded_files = get_pdf_files(directory_path)
                else:
                    st.warning("유효한 디렉토리 경로를 입력하세요.")
                    uploaded_files = None

            st.divider()

            openai_api_key = st.text_input(
                "OpenAI API Key를 입력하세요", key="chatbot_api_key", type="password"
            )
            process = st.button("OpenAI API Key 적용")
            # print(process)

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

    embedder = SentenceTransformer(
        "jhgan/ko-sroberta-multitask"
    )  # 한글을 잘 가져오는 거?
    if process:
        if not openai_api_key:
            st.info(
                "계속하려면 OpenAI API 키를 추가하세요."
            )  # Please add your OpenAI API key to continue.
            st.stop()

        modified_files = remove_underscore_after_pdf(uploaded_files)
        pdf_files_name = chg_itemname(modified_files)

        filename_embeddings = embedder.encode(pdf_files_name, convert_to_tensor=True)

        # filename_embeddings을 if query~에서도 사용하기 위해 session_state 사용
        st.session_state["embedded_filename"] = filename_embeddings

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

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        print("query", query)

        # template = """ 당신은 의학에 대해 대답하는 챗봇입니다. 주어진 문서에 있는 내용으로 대답하시오.
        # 만약 문서에 없는 내용이면 문서에 존재하지 않는다고 답변하세요. 모른다고 지어내서 말하지 말하세요.
        # File Name : 을 참조해서 어떤 절차에 대한 문서인지 파악하고 질문과 동일한 절차의 내용만 사용하세요.
        # {context}
        # 질문 : {query}
        # 너의 답변 : """

        # prompt = PromptTemplate(input_variables= ["context", "query"], template=template) -> !TODO prompt.메소드로 넣어야함 수정필요

        embedded_filename = st.session_state["embedded_filename"]

        # 질문과 유사한 파일을 고르고, 고른 파일만 가져옴
        question_embedding = embedder.encode(query, convert_to_tensor=True)
        # print(type(question_embedding), 'q')
        cos_scores = util.pytorch_cos_sim(question_embedding, embedded_filename).cpu()
        top_results = np.argpartition(-cos_scores, range(top_k))[
            0:top_k
        ]  # !TODO 단일 문서일 경우 예외처리는?

        files_to_copy = []
        for idx in top_results[0][0:top_k]:
            files_to_copy.append(uploaded_files[idx])

        copy_files_to_another_folder(origin_path, copy_path, files_to_copy)

        loader = PyPDFDirectoryLoader(copy_path)
        documents = loader.load()

        cleaned_documents = []
        for document in documents:
            cleaned_document = document  # metadata에서 파일명을 추출합니다.
            source_path = document.metadata.get("source", "")
            file_name = source_path.split("docs/")[-1]
            file_name = file_name.split("_")[0]
            file_name = unicodedata.normalize("NFC", file_name)

            page_content = document.page_content

            cleaned_text = f"File Name: {file_name} \n"

            # cleaned_text = preprocess_with_bert_embedding(page_content)  <- !TODO 사용시 정확도가 너무 낮음
            cleaned_text = page_content
            cleaned_document.page_content = cleaned_text
            cleaned_documents.append(cleaned_document)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(cleaned_documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            encode_kwargs={"normalize_embeddings": True},
        )

        vector_store = FAISS.from_documents(
            # vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
        )

        # 질문 받고 한글 전처리하고 모델 생성
        print("modeling ... ")
        st.session_state.conversation = get_conversation_chain(
            vector_store, openai_api_key, llm_option1, llm_option2
        )
        st.session_state.processComplete = True

        st.session_state.messages.append({"role": "user", "content": query})
        print("done !")

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
                    # st.markdown(
                    #     source_documents[1].metadata["source"],
                    #     help=source_documents[1].page_content,
                    # )
                    # st.markdown(
                    #     source_documents[2].metadata["source"],
                    #     help=source_documents[2].page_content,
                    # ) -> !TODO 참고문서 수를 1개로 줄였으니 1개만 나둠. top_k에 따른 예외처리 필요

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    top_k = 1  # 가져올 문서의 개수 설정
    DATA_PATH = "./datas/"  # 원본 파일 위치
    DOC_PATH = "./docs/"  # 복사한 파일 위치 (유사도순으로 추출 후)

    main(DATA_PATH, DOC_PATH, top_k)
    # BERT 토크나이저 로드
    # tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Q
# .doc 확장자는 어떻게?

# !TODO
# 템플릿으로 입력받기
# 파일 선택으로 1개 업로드시 예외처리
