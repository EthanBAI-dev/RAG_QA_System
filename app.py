import os
import re
import json
import html as html_mod
import streamlit as st
from RAG.VectorBase import VectorStore
from RAG.Embeddings import ZhipuEmbedding
from RAG.LLM import ZhipuAIChat, DeepSeekAIChat

st.set_page_config(page_title="RAG Q&A", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=Noto+Sans+JP:wght@400;500;700&display=swap');

:root {
  --ink: #1a1a2e;
  --ink-light: #3a3a4e;
  --paper: #faf8f5;
  --paper-white: #ffffff;
  --indigo-deep: #2d3a6e;
  --indigo-mid: #445e8c;
  --indigo-light: #e8edf5;
  --amber: #c8963e;
  --amber-light: #fef5e7;
  --green: #4a7c59;
  --green-light: #edf7f0;
  --border: #e8e4df;
  --shadow-sm: 0 1px 3px rgba(26,26,46,0.06);
  --shadow-md: 0 4px 24px rgba(26,26,46,0.08);
  --radius: 14px;
}

html, body, [class*="css"] {
  font-family: 'Zen Kaku Gothic New', 'Noto Sans JP', sans-serif !important;
  color: var(--ink);
}

.stApp {
  background: var(--paper);
}

.stMainBlockContainer {
  padding-top: 1.5rem !important;
}

.stSidebar {
  background: linear-gradient(180deg, var(--indigo-deep) 0%, #1e2a50 100%) !important;
}
.stSidebar .stCheckbox,
.stSidebar .stCheckbox label,
.stSidebar .stCheckbox span,
.stSidebar [data-testid="stCheckbox"] {
  color: rgba(255,255,255,0.88) !important;
  font-family: 'Zen Kaku Gothic New', 'Noto Sans JP', sans-serif !important;
}

.stSidebar [data-testid="stCheckbox"] label p {
  color: rgba(255,255,255,0.88) !important;
  font-size: 0.8rem !important;
}

.stSidebar .stMarkdown,
.stSidebar .stCaption,
.stSidebar label,
.stSidebar .stSlider > div > div > div {
  color: rgba(255,255,255,0.88) !important;
  font-family: 'Zen Kaku Gothic New', 'Noto Sans JP', sans-serif !important;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4,
.stSidebar .stMarkdown strong, .stSidebar .stMarkdown b {
  color: #ffffff !important;
}
.stSidebar [data-testid="stSelectbox"] label,
.stSidebar [data-testid="stSlider"] label,
.stSidebar [data-testid="stTextArea"] label {
  color: rgba(255,255,255,0.78) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.03em;
}
.stSidebar [data-baseweb="select"],
.stSidebar [data-baseweb="input"],
.stSidebar textarea {
  background: rgba(0,0,0,0.25) !important;
  border-color: rgba(255,255,255,0.18) !important;
  color: #ffffff !important;
}
.stSidebar [data-baseweb="select"] svg,
.stSidebar [data-baseweb="select"] path {
  fill: rgba(255,255,255,0.7) !important;
}
.stSidebar [data-baseweb="select"] [role="listbox"],
.stSidebar [data-baseweb="select"] ul {
  background: #1e2a50 !important;
  color: #ffffff !important;
}
.stSidebar [data-baseweb="select"] [role="listbox"] li,
.stSidebar [data-baseweb="select"] ul li {
  color: #ffffff !important;
}
.stSidebar [data-testid="stFileUploader"] section {
  background: rgba(0,0,0,0.2) !important;
  border: 1px dashed rgba(255,255,255,0.2) !important;
  border-radius: 8px !important;
}
.stSidebar [data-testid="stFileUploader"] section p,
.stSidebar [data-testid="stFileUploader"] section span,
.stSidebar [data-testid="stFileUploader"] section small {
  color: rgba(255,255,255,0.55) !important;
}
.stSidebar [data-testid="stFileUploader"] section button {
  background: rgba(255,255,255,0.12) !important;
  color: rgba(255,255,255,0.8) !important;
  border: 1px solid rgba(255,255,255,0.2) !important;
  border-radius: 6px !important;
}
.stSidebar .stButton > button {
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  color: #fff !important;
  border-radius: 8px !important;
  font-size: 0.8rem !important;
  transition: all 0.2s ease;
}
.stSidebar .stButton > button:hover {
  background: rgba(255,255,255,0.24) !important;
  border-color: rgba(255,255,255,0.4) !important;
}
.stSidebar .stButton > button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: rgba(255,255,255,0.45) !important;
  padding: 0 5px !important;
  font-size: 0.62rem !important;
  min-height: unset !important;
  height: 22px !important;
  line-height: 22px !important;
  border-radius: 4px !important;
}
.stSidebar .stButton > button[kind="secondary"]:hover {
  background: rgba(220,53,69,0.3) !important;
  border-color: rgba(220,53,69,0.5) !important;
  color: #ff6b6b !important;
}
.stSidebar [data-testid="stNotification"] {
  background: #1e2a50 !important;
}

.stChatMessage {
  background: var(--paper-white) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-sm) !important;
  padding: 16px 20px !important;
  margin-bottom: 12px !important;
}
.stChatMessage [data-testid="chatAvatarIcon-user"] {
  background: linear-gradient(135deg, var(--indigo-deep), var(--indigo-mid)) !important;
}
.stChatMessage [data-testid="chatAvatarIcon-assistant"] {
  background: linear-gradient(135deg, var(--amber), #d4a84c) !important;
}

@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
.stChatMessage {
  animation: fadeSlideIn 0.4s ease-out;
}

.stChatMessage:nth-child(odd) {
  animation-delay: 0.05s;
}

.stTextInput > div > div > input,
.stChatInput textarea {
  border-radius: 10px !important;
  border: 1.5px solid var(--border) !important;
  background: var(--paper-white) !important;
  padding: 12px 16px !important;
  font-size: 0.95rem !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stChatInput textarea:focus,
.stTextInput > div > div > input:focus {
  border-color: var(--indigo-mid) !important;
  box-shadow: 0 0 0 3px rgba(68,94,140,0.12) !important;
}

.stSelectbox [data-baseweb="select"] > div {
  border-radius: 8px !important;
  border-color: var(--border) !important;
}

hr, .stDivider {
  border-color: var(--border) !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #aaa; }

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
.skeleton {
  background: linear-gradient(90deg, #f0ece6 25%, #e8e4df 50%, #f0ece6 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
  border-radius: var(--radius);
}

@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(68,94,140,0.3); }
  50% { box-shadow: 0 0 0 8px rgba(68,94,140,0); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 8px 0 0 0; margin-bottom: 4px;">
  <div style="display: flex; align-items: baseline; gap: 16px;">
    <div style="font-size: 1.55rem; font-weight: 700; color: var(--ink); letter-spacing: -0.01em;">
      RAG Q&A
    </div>
    <div style="font-size: 0.85rem; color: #888; font-weight: 400; letter-spacing: 0.04em;">
      AI-powered document intelligence
    </div>
  </div>
  <div style="margin-top: 4px; width: 48px; height: 3px; background: linear-gradient(90deg, var(--amber), #e0bc6c); border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)


def get_document_summaries(data_dir='./data'):
    summaries = []
    if not os.path.isdir(data_dir):
        return summaries
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(('.md', '.txt', '.pdf')):
            continue
        filepath = os.path.join(data_dir, f)
        try:
            if f.endswith('.pdf'):
                import PyPDF2
                with open(filepath, 'rb') as pf:
                    reader = PyPDF2.PdfReader(pf)
                    text = reader.pages[0].extract_text() if reader.pages else ""
            else:
                with open(filepath, 'r', encoding='utf-8') as mf:
                    text = mf.read()
            text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n+', ' ', text).strip()
            summary = text[:120] + ('…' if len(text) > 120 else '')
            summaries.append((f, summary))
        except Exception:
            summaries.append((f, ''))
    return summaries


def get_sample_questions(document_summaries):
    default_questions = [
        "このドキュメントの概要を教えてください",
        "安全上の注意点は何ですか？",
        "使い方の手順を説明してください",
        "メンテナンス方法を教えて",
    ]
    if not document_summaries:
        return default_questions

    fnames = [fname for fname, _ in document_summaries]

    questions = []
    for fname in fnames[:4]:
        base_name = re.sub(r'\.[^.]+$', '', fname)
        target = re.sub(r'[_：:].*$', '', base_name)
        match = re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,8}', target)
        keyword = match.group(0) if match else target[:8]
        questions.append(f"「{keyword}」について教えて")

    if len(questions) < 4:
        for q in default_questions:
            if len(questions) >= 4:
                break
            if q not in questions:
                questions.append(q)

    return questions[:4]

DEFAULT_FILES = {'ひざ電器治療バンド.md', '低周波治療器.md'}

MODEL_OPTIONS = {
    "DeepSeek-V4": "deepseek-chat",
    "DeepSeek-Flash": "deepseek-chat",
}

MODEL_PROVIDER = {
    "DeepSeek-V4": "deepseek",
    "DeepSeek-Flash": "deepseek",
}

ANSWER_STYLES = {
    "概要レポート": "概要レポート",
    "学習ガイド": "学習ガイド",
    "ブログ記事": "ブログ記事",
    "カスタム形式": "カスタム形式",
}

LENGTH_OPTIONS = [200, 400, 800]


def strip_html(raw: str) -> str:
    text = html_mod.unescape(raw)
    text = re.sub(r'<[^>]*>', '', text, flags=re.DOTALL)
    return text


def highlight_text(text, query):
    if not query:
        return text
        
    clean_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    eng_num = re.findall(r'[a-zA-Z0-9]{2,}', query)
    
    cjk_pure = re.sub(r'[^\u4e00-\u9fa5\u3040-\u30ff]', '', query)
    
    stopwords = ["的", "是", "在", "了", "和", "与", "有", "就", "不", "都", "一", "什么", "怎么", "如何"]
    for sw in stopwords:
        cjk_pure = cjk_pure.replace(sw, "")
        
    cjk_words = []
    if len(cjk_pure) >= 2:
        for length in [4, 3, 2]:
            for i in range(len(cjk_pure) - length + 1):
                cjk_words.append(cjk_pure[i:i+length])
    elif len(cjk_pure) == 1:
        cjk_words.append(cjk_pure)
        
    keywords = list(set(eng_num + cjk_words))
    
    if not keywords:
        return clean_text
        
    keywords.sort(key=len, reverse=True)
    
    pattern = re.compile(f"({'|'.join(map(re.escape, keywords))})", flags=re.IGNORECASE)
    parts = pattern.split(clean_text)
    
    result = ''
    for part in parts:
        if part and pattern.fullmatch(part):
            result += f'<span style="color:#ff4b4b;font-weight:bold">{part}</span>'
        else:
            result += part
            
    return result


def _load_build_manifest(storage_dir):
    manifest_path = f'{storage_dir}/build_manifest.json'
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def _save_build_manifest(storage_dir, manifest):
    manifest_path = f'{storage_dir}/build_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f)


@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    from RAG.utils import ReadFiles

    data_dir = './data'
    storage_dir = './storage'
    vectors_path = f'{storage_dir}/vectors.json'
    embedding = ZhipuEmbedding()

    os.makedirs(storage_dir, exist_ok=True)

    data_files = {f: os.path.getmtime(os.path.join(data_dir, f))
                  for f in os.listdir(data_dir)
                  if f.endswith(('.md', '.txt', '.pdf')) and os.path.isfile(os.path.join(data_dir, f))}

    manifest = _load_build_manifest(storage_dir)

    if not os.path.exists(vectors_path):
        docs, sources = ReadFiles(data_dir).get_content(max_token_len=600, cover_content=150)
        vector = VectorStore(docs, sources)
        vector.get_vector(EmbeddingModel=embedding)
        vector.persist(path=storage_dir)
        manifest = {f: data_files[f] for f in data_files}
        _save_build_manifest(storage_dir, manifest)
        from collections import Counter
        source_counts = Counter(sources)
        file_items = ''.join(
            f'<li style="font-size:14px; color:#28a745;">{name}: <b>{cnt}</b> チャンク</li>'
            for name, cnt in source_counts.items()
        )
        st.markdown(
            f'<div style="border:1px solid #c3e6cb; border-radius:6px; padding:12px 16px; background:#f0fff4;">'
            f'<details>'
            f'<summary style="cursor:pointer; color:#28a745; font-weight:600; font-size:15px;">'
            f'ベクトル構築が完了しました！ 合計 {len(docs)} チャンク ／ {len(source_counts)} ファイル'
            f'</summary>'
            f'<ul style="margin-top:8px; line-height:1.8; color:#28a745;">{file_items}</ul>'
            f'</details>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return vector, embedding

    vector = VectorStore()
    vector.load_vector(storage_dir)

    deleted_files = [f for f in manifest if f not in data_files]
    for df in deleted_files:
        vector.remove_by_source(df, persist_path=None)
        del manifest[df]

    new_or_changed = []
    for fname, fmtime in data_files.items():
        if fname not in manifest or manifest[fname] != fmtime:
            new_or_changed.append(fname)

    if new_or_changed or deleted_files:
        details = []
        if deleted_files:
            details.append(f'{len(deleted_files)}件の削除')
        if new_or_changed:
            details.append(f'{len(new_or_changed)}件の追加/更新: {", ".join(new_or_changed)}')

        for fname in new_or_changed:
            vector.remove_by_source(fname, persist_path=None)
            tmp_dir = os.path.join(storage_dir, '_tmp_input')
            os.makedirs(tmp_dir, exist_ok=True)
            src = os.path.join(data_dir, fname)
            dst = os.path.join(tmp_dir, fname)
            import shutil
            shutil.copy2(src, dst)

            new_docs, new_sources = ReadFiles(tmp_dir).get_content(max_token_len=600, cover_content=150)
            if not new_docs:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            new_vec = VectorStore(new_docs, new_sources)
            new_vec.get_vector(EmbeddingModel=embedding)
            vector.append(new_vec.document, new_vec.sources, new_vec.vectors, persist_path=None)

            manifest[fname] = fmtime
            shutil.rmtree(tmp_dir, ignore_errors=True)

        vector.persist(path=storage_dir)
        _save_build_manifest(storage_dir, manifest)
        from collections import Counter
        source_counts = Counter(vector.sources)
        file_items = ''.join(
            f'<li style="font-size:14px; color:#28a745;">{name}: <b>{cnt}</b> チャンク</li>'
            for name, cnt in source_counts.items()
        )
        st.markdown(
            f'<div style="border:1px solid #c3e6cb; border-radius:6px; padding:12px 16px; background:#f0fff4;">'
            f'<details>'
            f'<summary style="cursor:pointer; color:#28a745; font-weight:600; font-size:15px;">'
            f'増分構築が完了しました！ 合計 {len(vector.document)} チャンク ／ {len(source_counts)} ファイル'
            f'</summary>'
            f'<ul style="margin-top:8px; line-height:1.8; color:#28a745;">{file_items}</ul>'
            f'</details>'
            f'</div>',
            unsafe_allow_html=True,
        )

    return vector, embedding


def get_chat_model(model_name: str):
    provider = MODEL_PROVIDER.get(model_name, "deepseek")
    model_id = MODEL_OPTIONS.get(model_name, "deepseek-chat")

    if provider == "deepseek":
        return DeepSeekAIChat(model=model_id)

    return DeepSeekAIChat(model=model_id)

if "history" not in st.session_state:
    st.session_state.history = []

if "context_docs" not in st.session_state:
    st.session_state.context_docs = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "DeepSeek-V4"

if "answer_style" not in st.session_state:
    st.session_state.answer_style = "概要レポート"

if "answer_length" not in st.session_state:
    st.session_state.answer_length = 400

if "custom_instruction" not in st.session_state:
    st.session_state.custom_instruction = ""

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "file_uploader_counter" not in st.session_state:
    st.session_state.file_uploader_counter = 0

doc_summaries = get_document_summaries('./data')

with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 2px 0;">
      <div style="font-size: 1.1rem; font-weight: 700; letter-spacing: 0.02em; margin-bottom: 2px;">ナレッジベース</div>
      <div style="font-size: 0.68rem; opacity: 0.6; letter-spacing: 0.05em;">KNOWLEDGE BASE</div>
    </div>
    """, unsafe_allow_html=True)

    data_dir = './data'
    if os.path.isdir(data_dir):
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.md', '.txt', '.pdf'))])
        if files:
            for f in files:
                c_fname, c_del = st.columns([9, 1], vertical_alignment="center")
                with c_fname:
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:6px;
                    padding:0 8px; margin:0; border-radius:5px;
                    background:rgba(255,255,255,0.06); font-size:0.75rem;
                    height:40px; line-height:40px;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    <span style="opacity:0.45; font-size:0.7rem;">📄</span>
                    <span>{f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with c_del:
                    if st.button("✕", key=f"sidebar_del_{f}", help=f"{f} を削除", type="secondary"):
                        os.remove(os.path.join(data_dir, f))
                        load_knowledge_base.clear()
                        st.rerun()
        else:
            st.caption("資料がありません")
    else:
        st.caption("資料ディレクトリが存在しません")

    data_dir_for_upload = './data'
    os.makedirs(data_dir_for_upload, exist_ok=True)

    st.markdown("""
    <div style="margin-top:12px; margin-bottom:4px; font-size:0.72rem; opacity:0.55; letter-spacing:0.04em;">
    ファイルを追加</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
        key=f"sidebar_uploader_{st.session_state.file_uploader_counter}",
        label_visibility="collapsed",
    )

    if uploaded:
        for uf in uploaded:
            filepath = os.path.join(data_dir_for_upload, uf.name)
            with open(filepath, 'wb') as wf:
                wf.write(uf.getbuffer())
        st.session_state.file_uploader_counter += 1
        load_knowledge_base.clear()
        st.rerun()

    st.divider()

    st.markdown("""
    <div style="margin-bottom:6px;">
      <div style="font-size: 1.1rem; font-weight: 700; letter-spacing: 0.02em; margin-bottom: 2px;">回答設定</div>
      <div style="font-size: 0.68rem; opacity: 0.6; letter-spacing: 0.05em;">RESPONSE SETTINGS</div>
    </div>
    """, unsafe_allow_html=True)

    st.selectbox(
        "LLMモデル",
        options=list(MODEL_OPTIONS.keys()),
        key="model_name",
    )

    st.selectbox(
        "回答の長さ",
        options=LENGTH_OPTIONS,
        key="answer_length",
        format_func=lambda x: f"{x} トークン",
    )

    st.selectbox(
        "回答スタイル",
        options=list(ANSWER_STYLES.keys()),
        key="answer_style",
    )

    if st.session_state.answer_style == "カスタム形式":
        st.text_area(
            "カスタム指示",
            key="custom_instruction",
            placeholder="例：箇条書きで簡潔に答えてください。",
        )

    k = st.slider("検索チャンク数 Top-K", 1, 5, 3)

    show_context = st.checkbox("検索内容を表示", True)

    if st.button("🗑 会話をクリア", use_container_width=True):
        st.session_state.history = []
        st.session_state.context_docs = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("""
    <div style="position:fixed; bottom:16px; font-size:0.62rem; opacity:0.35; text-align:center; width:calc(100% - 48px);">
    RAG Q&A System v2.0<br>&copy; 2026
    </div>
    """, unsafe_allow_html=True)

if doc_summaries:
    file_items = ''.join(
        f'<div style="display:flex; align-items:baseline; gap:8px; margin-top:6px; font-size:12px;">'
        f'<span style="color:var(--indigo-mid); font-weight:600; white-space:nowrap;">{fname}</span>'
        f'<span style="color:#888; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{summary}</span>'
        f'</div>'
        for fname, summary in doc_summaries
    )
    st.markdown(
        f'<div style="background:var(--paper-white); border:1px solid var(--border); '
        f'border-radius:var(--radius); padding:14px 20px; margin-bottom:8px; '
        f'box-shadow:var(--shadow-sm);">'
        f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">'
        f'<div style="width:6px; height:6px; border-radius:50%; background:var(--amber);"></div>'
        f'<span style="font-size:13px; font-weight:700; color:var(--ink); letter-spacing:0.01em;">'
        f'ナレッジベース</span>'
        f'<span style="font-size:11px; color:#aaa; margin-left:auto;">{len(doc_summaries)} docs</span>'
        f'</div>'
        f'{file_items}'
        f'</div>',
        unsafe_allow_html=True,
    )

loading_html = """
<div style="display:flex; justify-content:center; align-items:center; min-height:200px;">
  <div style="text-align:center;">
    <div style="width:40px; height:40px; margin:0 auto 16px auto; border-radius:50%;
    border:3px solid var(--border); border-top-color:var(--indigo-mid);
    animation: spin 0.8s linear infinite;"></div>
    <div style="font-size:15px; font-weight:600; color:var(--ink);">ナレッジベースを読み込み中...</div>
    <div style="font-size:12px; color:#aaa; margin-top:4px;">初回はベクトル構築のため数十秒かかります</div>
  </div>
</div>
<style>
@keyframes spin { to { transform: rotate(360deg); } }
</style>
"""
loading_placeholder = st.empty()
loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
vector, embedding = load_knowledge_base()
loading_placeholder.empty()

st.markdown('<hr style="margin:12px 0 8px 0;border:none;border-top:1px solid var(--border);">', unsafe_allow_html=True)

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

sample_questions = get_sample_questions(doc_summaries)
if sample_questions:
    st.markdown(
        '<p style="font-size:12px; color:#aaa; margin:10px 0 8px 0; letter-spacing:0.03em;">'
        '質問例をクリック</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(sample_questions))
    for i, (col, q) in enumerate(zip(cols, sample_questions)):
        with col:
            st.markdown(f"""
            <style>
            div[data-testid="stButton"] button[kind="secondary"][id="sample_{st.session_state.file_uploader_counter}_{i}"] {{
              background: var(--paper-white) !important;
              border: 1px solid var(--border) !important;
              border-radius: 20px !important;
              font-size: 0.78rem !important;
              padding: 6px 14px !important;
              color: var(--ink-light) !important;
              transition: all 0.2s ease;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }}
            div[data-testid="stButton"] button[kind="secondary"][id="sample_{st.session_state.file_uploader_counter}_{i}"]:hover {{
              background: var(--indigo-light) !important;
              border-color: var(--indigo-mid) !important;
              color: var(--indigo-deep) !important;
            }}
            </style>
            """, unsafe_allow_html=True)
            if st.button(
                q, key=f"sample_q_{st.session_state.file_uploader_counter}_{i}",
                use_container_width=True, type="secondary",
            ):
                st.session_state.pending_question = q
                st.rerun()

question = st.chat_input("ドキュメントについて質問してください...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            try:
                docs = vector.query(question, EmbeddingModel=embedding, k=k)
                st.session_state.context_docs = docs
                context = "\n".join([d["text"] for d in docs])

                style = st.session_state.answer_style
                custom_instruction = ""
                if style == "カスタム形式" and st.session_state.get("custom_instruction", "").strip():
                    custom_instruction = st.session_state.custom_instruction
                max_tokens = st.session_state.answer_length

                chat = get_chat_model(st.session_state.model_name)
                answer = st.write_stream(chat.chat_stream(
                    question, [], context,
                    style=style,
                    custom_instruction=custom_instruction,
                    max_tokens=max_tokens,
                ))

            except Exception as e:
                answer = f"システムエラーが発生しました。\n\nエラー詳細：{e}"
                st.write(answer)

        st.session_state.history.append(("assistant", answer))

if show_context and st.session_state.context_docs:
    with st.expander("検索内容（Top-K）", expanded=False):
        last_question = ""
        if len(st.session_state.history) >= 2:
            last_question = st.session_state.history[-2][1]
        elif len(st.session_state.history) == 1:
            last_question = st.session_state.history[0][1]

        for i, doc in enumerate(st.session_state.context_docs):
            text = doc["text"]
            score = doc.get("score", 0)
            source = doc.get("source", "")

            text_clean = strip_html(text)

            st.markdown(f"### Chunk {i+1}")
            st.caption(f"出典: {source}  |  類似度: {score:.4f}")

            highlighted = highlight_text(text_clean, last_question)
            
            html_content = (
                f'<div style="border:1px solid #ddd; padding:15px; border-radius:10px; margin-bottom:15px; line-height:1.8;">'
                f'{highlighted}'
                f'</div>'
            )
            
            st.markdown(html_content, unsafe_allow_html=True)
