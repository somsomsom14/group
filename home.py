



import streamlit as st
st.set_page_config(page_title="강남대 챗봇 시작", page_icon="🐑")


st.markdown(
    """
    <style>
        body {
            background-color: #e3f3fc; 
        }
        .stApp {
            background-color: #e3f3fc; 
        }
    </style>
    """,
    unsafe_allow_html=True
)



LOGO_URL_LARGE = 'https://i.namu.wiki/i/23lxs6whoK9vB9vMuDk71QF94SKZ6vEonEyzzQoy9mHIOdI9IbdyTXu-bkXjFUzqa0CxxgyeJhgr_ce86Z4QRxXXadDI0MzMF6U4mZXyEI4.svg'

st.logo(
    LOGO_URL_LARGE,
    link="https://web.kangnam.ac.kr/index.do",
    size="large"
)


st.markdown("""
<h2 style='text-align: center;'>👋 강남대 챗봇에 오신 걸 환영합니다!</h2>
""", unsafe_allow_html=True)


# 가운데 정렬을 위해 3개의 컬럼 생성
col1, col2, col3 = st.columns([1, 2, 1])


with col2:
    st.image(r"C:\\Users\\최소미\\Desktop\\str\\image\\ramb.png", width=800)

st.markdown("""
<div style="
    background-color:white;
    padding:15px;
    border-radius:10px;
    box-shadow:0px 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    text-align: center;  /* 가운데 정렬 */
">
 안녕하세요!<br>
 저는 강남대의 <span style='font-size:18px; font-weight:bold; color:#8abbda;'>⭐️마스코트 람브⭐️</span> 라고 해요<br>
 궁금한 게 있다면 뭐든지 물어보세요!<br>
</div>
""", unsafe_allow_html=True)



st.markdown("""
<div style="text-align:center; margin-top:25px;">  
    <a href="/gangnam" target="_self" style="text-decoration: none;">
        <button style="
            background-color:#8abbda;
            color: white;
            padding: 0.5em 1.0em;
            font-size: 1.2em;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        ">
            ▶️ 챗봇 페이지로 가기
        </button>
    </a>
</div>
""", unsafe_allow_html=True)
