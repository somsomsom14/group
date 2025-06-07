import pdfplumber
import os
import streamlit as st
from dotenv import load_dotenv
import logging
# 문서 로딩 및 처리
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# OpenAI 및 벡터 저장소
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# LangChain 핵심 체인 관련
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

# 프롬프트 구성
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Streamlit용 대화 기록
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# 멀티 쿼리 리트리버
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# 📄 PDF 문서 불러오기

def load_pdf_with_tables(directory_path):
    docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(directory_path, filename)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()

                    table_texts = []
                    for table in tables:
                        rows = ["\t".join(cell if cell is not None else "" for cell in row) for row in table if row]
                        table_texts.append("\n".join(rows))

                    full_text = text + "\n\n" + "\n\n".join(table_texts)
                    docs.append(Document(page_content=full_text, metadata={"source": filename, "page": page_num + 1}))
    return docs


@st.cache_resource
def load_pdf():
    file_path = "./dat" 
    return load_pdf_with_tables(file_path)



# 🧠 벡터스토어 초기화 또는 로드
@st.cache_resource
def get_or_create_vectorstore():
    persist_directory = "./faissDb"
    embedding = OpenAIEmbeddings(model='text-embedding-3-small')

    if os.path.exists(os.path.join(persist_directory, "index.faiss")):
        return FAISS.load_local(persist_directory, embedding, allow_dangerous_deserialization=True)

    docs = load_pdf()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embedding)
    vectorstore.save_local(persist_directory)
    return vectorstore

# 📑 문서 포맷터
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 🔗 체인 구성
@st.cache_resource
def chaining():
    vectorstore = get_or_create_vectorstore()
    
    # ✅ MultiQueryRetriever로 교체
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    from langchain.retrievers.multi_query import MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    # ✅ 로그 출력 (원한다면 유지)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    # 🔁 히스토리 기반 질문 리포맷
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 🧠 답변 프롬프트
    qa_system_prompt = """
    You are 람브, a friendly and knowledgeable assistant who helps new university students.  
Always answer in Korean, using warm, polite, and easy-to-understand language. 😊  
Include appropriate emojis to make your responses feel approachable and friendly. 🌟

Use the provided documents as your primary reference.  
When answering, include **all relevant information from the documents** without omitting any helpful detail.

If you also know accurate and relevant information beyond the documents, you may include it to help the user.  
You do not need to mention whether it came from the document or not.

If the information is not known to you, respond politely with:  
“죄송해요, 그 부분은 문서에 나와 있지 않아요.”  

If the user asks for a specific form or template:
1. Explain the purpose and usage of the form based on its name or contents.
2. If available, offer the file with a friendly download message.
3. If not available, let the user know kindly.

If the question is ambiguous, ask the user for clarification.  
Your goal is to be kind, clear, and as informative as possible using both documents and your own knowledge. 📚

    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 🔁 체인 구성
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

TEMPLATE_MAP = {
    "단체": ["단체등록신청서.docx"],
    "논문": ["논문 작성 계획서.docx"],
    "봉사": ["교육봉사활동승인신청서.docx", "교육봉사활동확인서.docx", '교육봉사활동 일지.hwp'],
    "실습": ["실습신청서.docx", "실습보고서 작성 계획서.docx",
           '학교현장실습지도 안내 및 출근부, 결과보고서(양식)_유치원.hwp',
           '학교현장실습지도 안내 및 출근부, 결과보고서(양식)_중등학교.hwp',
           '학교현장실습지도 안내 및 출근부, 결과보고서(양식)_특수학교.hwp',
           '개인정보 수집.이용. 제3자 제공 동의서.hwp',
           '2025_학교현장실습일지(유치원).hwp',
           '2025_학교현장실습일지(중등학교).hwp',
           '2025_학교현장실습일지(특수하교).hwp',
           "학교현장실습결과보고서.docx"],
    "인쇄": ["인쇄물배포원.docx"],
    "교과": ["교직과목 학점인정서.docx", "교직과정(복수전공)이수신청서.docx"],
    "평생교육": [
        "평생교육사 자격증 발급 신청서.hwp",
        "평생교육사 자격증 재발급신청서.docx",
        "평생교육 현장실습 평가서.docx",
        '(예시)평생교육사 자격증 발급 신청서.pdf'
    ],
    "졸업": ["조기졸업신청서.docx", "졸업유보신청서.docx", "졸업논문계획서.docx"],
    "활동": ["활동계획 및 예산서.docx"],
    "학생": ["학생교류지원서.docx"],
    "학점": ["학점교류지원서.docx"],
    "자격증": ["학점교류지원서.docx",
            '영문교원자격증발급신청서.hwp',
            '사서자격증발급신청서 양식.hwp',
            '교원자격증 재교부 신청서(양식).hwp'],
    '교직': ['2025학년도 입학자 교직이수 가이드.hwp']
}
def match_template_keywords(message):
    if not message:
        return []
    
    trigger_words = ["양식", "신청서", "서식", "다운로드", "폼", "서류", "양식들", "양식좀", "양식요", "폼좀"]
    return [
        k for k in TEMPLATE_MAP
        if k in message and any(trigger in message for trigger in trigger_words)
    ]



st.markdown(
    """
    <style>
        /* 전체 배경 */
        .stApp {
            background-color: #e3f3fc;
        }
    </style>
    """,
    unsafe_allow_html=True
)




# 🖼️ 앱 인터페이스 시작
st.header("강남대 챗봇 🐑")

rag_chain = chaining()
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 1. FAQ 버튼 클릭 처리
st.markdown("""
    <h5 style="color:#333333; font-weight:bold;">자주 묻는 질문🧐</h5>
""", unsafe_allow_html=True)

faq_list = [
    "강남대 순환버스 시간표를 알려줘",
    "강남대 졸업 조건 알려줘",
    "교학1팀 전화번호 알려줘"
]

page_prefix = "gangnam"

for idx, faq in enumerate(faq_list):
    if st.button(f"{faq}", key=f"{page_prefix}_faq_{idx}", type="secondary"):
        st.session_state["faq_question"] = faq


if "docs_with_scores" not in st.session_state:
    st.session_state.docs_with_scores = []

if "download_list" not in st.session_state:
    st.session_state.download_list = []
    
# 2. 초기 메시지 설정
if "profile_message" not in st.session_state:
    st.session_state["profile_message"] = {
        "role": "assistant",
        "content": "강남대에 대해 물어보면, 람브가 알려줄게요!🐑"
    }

if "messages" not in st.session_state:
    st.session_state["messages"] = []



# 3. 기존 대화 렌더링
profile = st.session_state["profile_message"]
with st.chat_message("assistant", avatar="./image/chat_ramb.jpg"):
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff; padding: 10px; border-radius: 10px;">
            📣 {profile["content"]}
        </div>
        """,
        unsafe_allow_html=True
    )


for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant",avatar="./image/chat_ramb.jpg"):
            st.markdown(
                f"""
                <div style="background-color:#f0f8ff; padding: 15px; border-radius: 12px; font-size: 15px;">
                    {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )


        if "download_list" in msg:
            st.divider()
            for info in msg["download_list"]:
                filename = info["filename"]
                file_bytes = info["file_bytes"]
                mime_type = {
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ".pdf": "application/pdf",
                    ".hwp": "application/octet-stream"
                }.get(filename[filename.rfind("."):], "application/octet-stream")

                st.info(f"📝 파일 다운로드: {filename}")
                st.download_button(
                    label=f"📄 {filename} 다운로드",
                    data=file_bytes,
                    file_name=filename,
                    mime=mime_type,
                    key=filename
                )
    else:
        st.chat_message("user").write(msg["content"])

# 4. FAQ 질문 처리 or 일반 입력 처리
prompt_message = None
from_faq = False

if "faq_question" in st.session_state:
    prompt_message = st.session_state.pop("faq_question")
    from_faq = True  # <- FAQ에서 왔다는 플래그 설정
    st.chat_message("user").write(prompt_message)

user_input = st.chat_input("제출 양식과 강남대 정보 등을 제공합니다!", key="gangnam_chat_input")

if user_input and not from_faq:
    prompt_message = user_input

if prompt_message:
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    if not from_faq:
        st.chat_message("user").write(prompt_message)

    with st.spinner("답변 준비 중..."):
        response = conversational_rag_chain.invoke(
            {"input": prompt_message},
            {"configurable": {"session_id": "any"}}
        )
        answer = response["answer"]

        vectorstore = get_or_create_vectorstore()
        docs_with_scores = vectorstore.similarity_search_with_score(prompt_message, k=3)

        download_list = []
        matched_keywords = match_template_keywords(prompt_message)
        for keyword in matched_keywords:
            filenames = TEMPLATE_MAP.get(keyword, [])
            for filename in filenames:
                file_path = os.path.join("./templates", filename)
                try:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    answer += f"\n\n📎 참고로, '{filename}' 양식은 다운로드할 수 있어요! 👇"
                    download_list.append({"filename": filename, "file_bytes": file_bytes})
                except FileNotFoundError:
                    answer += f"\n\n⚠️ '{filename}' 파일이 존재하지 않아요. 관리자에게 문의해주세요!"

        assistant_msg = {"role": "assistant", "content": answer}
        if download_list:
            assistant_msg["download_list"] = download_list

        # 메시지에 다운로드 리스트 포함해 저장해야 UI 렌더링 시 읽을 수 있음
        st.session_state.messages.append(assistant_msg)

        # 세션 상태에 참고 문서, 다운로드 리스트 저장
        st.session_state.docs_with_scores = docs_with_scores
        st.session_state.download_list = download_list

    # st.rerun() 호출하지 말고, 자연스럽게 다음 렌더링으로 넘어가도록


# 2) UI 렌더링할 때
    with st.chat_message("assistant", avatar="./image/chat_ramb.jpg"):
        st.markdown(
            f"""
            <div style="background-color:#f0f8ff; padding: 15px; border-radius: 12px; font-size: 15px;">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.expander("📂 참고 문서 확인"):
            for doc, score in st.session_state.docs_with_scores:
                filename = os.path.basename(doc.metadata["source"])
                st.markdown(f"📄 **{filename}** &nbsp;&nbsp;&nbsp; *(score: {score:.4f})*", help=doc.page_content)
        if st.session_state.download_list:
            st.divider()
            for info in st.session_state.download_list:
                st.info(f"📝 파일 다운로드: {info['filename']}")
                st.download_button(
                    label=f"📄 {info['filename']} 다운로드",
                    data=info["file_bytes"],
                    file_name=info["filename"],
                    mime=(
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        if info["filename"].endswith(".docx")
                        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ),
                    key=info["filename"])