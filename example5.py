import streamlit as st
import tiktoken
from operator import itemgetter
from loguru import logger

from langchain_core.messages import ChatMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langserve import RemoteRunnable


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
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def main():
    global retriever

    st.set_page_config(
        page_title="Streamlit_remote_RAG",
        page_icon=":books:"
    )

    st.title("_RAG_test4 :red[Q/A Chat]_ :books:")

    # 화면 출력용 대화 기록 초기화 (ChatMessage 객체로 통일)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            ChatMessage(role="assistant", content="안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!")
        ]

    # LLM 기억용 세션 대화 기록 저장소 초기화
    if "store" not in st.session_state:
        st.session_state["store"] = dict()

    # 세션 ID별 대화 히스토리를 가져오는 함수
    def get_session_history(session_ids):
        if session_ids not in st.session_state["store"]:
            st.session_state["store"][session_ids] = ChatMessageHistory()
        return st.session_state["store"][session_ids]

    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")

    if process and uploaded_files:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        retriever = vectorstore.as_retriever(search_type='mmr', verbose=True)  # 원본 오타(vervose) 수정
        st.session_state['retriever'] = retriever
        st.session_state.processComplete = True

    def format_docs(docs):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)

    # 이전 대화 기록(chat_history)도 참고할 수 있도록 프롬프트 구성
    RAG_PROMPT_TEMPLATE = """당신은 동서울대학교 컴퓨터소프트웨어과 안내 AI 입니다.
검색된 문맥(Context)과 이전 대화 기록(Chat History)을 사용하여 질문에 맞는 답변을 30단어 이내로 하세요.
답을 모른다면 모른다고 답변하세요.

Context: {context}"""

    print_history()

    if user_input := st.chat_input("메세지를 입력해 주세요"):
        # 사용자가 입력한 내용 저장 및 출력
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")

        with st.chat_message("assistant"):
            # ※ 본인의 Colab/로컬 LangServe ngrok 주소로 변경해서 사용하세요.
            llm = RemoteRunnable("https://centrally-exciting-liger.ngrok-free.app/llm/")
            chat_container = st.empty()

            # 1. 문서 업로드 후 RAG 시스템이 동작할 때
            if st.session_state.processComplete == True:
                prompt1 = ChatPromptTemplate.from_messages([
                    ("system", RAG_PROMPT_TEMPLATE),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ])

                retriever = st.session_state['retriever']

                # 히스토리 딕셔너리 입력에서 question만 뽑아 retriever에 전달
                rag_chain = (
                    {
                        "context": itemgetter("question") | retriever | format_docs,
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt1
                    | llm
                    | StrOutputParser()
                )

                rag_with_history = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="question",
                    history_messages_key="chat_history",
                )

                answer = rag_with_history.stream(
                    {"question": user_input},
                    config={"configurable": {"session_id": "default_session"}},
                )

                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown("".join(chunks))
                add_history("ai", "".join(chunks))

            # 2. 문서를 업로드하지 않고 일반 LLM으로 대화할 때
            else:
                prompt2 = ChatPromptTemplate.from_messages([
                    ("system", "다음의 질문에 이전 대화 맥락을 참고하여 간결하게 답변해 주세요."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                chain = prompt2 | llm | StrOutputParser()

                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                )

                answer = chain_with_history.stream(
                    {"input": user_input},
                    config={"configurable": {"session_id": "default_session"}},
                )

                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown("".join(chunks))
                add_history("ai", "".join(chunks))


if __name__ == '__main__':
    main()