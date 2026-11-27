import streamlit as st
import tiktoken
from loguru import logger

from langchain_core.messages import ChatMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langserve import RemoteRunnable


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    st.set_page_config(
        page_title="RAG_test4",
        page_icon=":books:"
    )

    st.title("_RAG_test4 :red[Q/A Chat]_ :books:")

    if "store" not in st.session_state:
        st.session_state.store = {}

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}
        ]

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")

    if process:
        if not uploaded_files:
            st.warning("파일을 먼저 업로드해주세요.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
        st.session_state.retriever = vetorestore.as_retriever(search_type='mmr', verbose=True)
        st.session_state.processComplete = True
        st.success("문서 처리가 완료되었습니다!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def get_session_history(session_ids: str) -> BaseChatMessageHistory:
        if session_ids not in st.session_state.store:
            st.session_state.store[session_ids] = ChatMessageHistory()
        return st.session_state.store[session_ids]

    # 본인의 Ngrok 고정 도메인 적용 완료
    llm = RemoteRunnable("https://extenuate-herbs-alone.ngrok-free.dev/llm/")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if st.session_state.processComplete:
                retriever = st.session_state.retriever
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "당신은 친절한 AI 어시스턴트입니다. 주어진 문맥(Context)을 바탕으로 질문에 정확하게 한국어로 답변하세요.\n\nContext:\n{context}"),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ])
                chain = (
                    {
                        "context": lambda x: format_docs(retriever.invoke(x["question"])),
                        "question": lambda x: x["question"],
                        "history": lambda x: x["history"],
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "당신은 친절한 AI 어시스턴트입니다. 사용자의 질문에 한국어로 답변하세요."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ])
                chain = prompt | llm | StrOutputParser()

            chain_with_memory = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )

            response_placeholder = st.empty()
            full_response = ""
            answer = chain_with_memory.stream(
                {"question": query},
                config={"configurable": {"session_id": "user_session"}}
            )
            for chunk in answer:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == '__main__':
    main()
