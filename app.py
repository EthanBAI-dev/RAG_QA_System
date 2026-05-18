import os
import re
import json
import html as html_mod
import streamlit as st
from RAG.VectorBase import VectorStore
from RAG.Embeddings import ZhipuEmbedding
from RAG.LLM import ZhipuAIChat, DeepSeekAIChat

st.set_page_config(page_title="RAGシステム", layout="wide")
st.title("RAGインテリジェントQ&Aシステム")


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
        if len(base_name) > 30:
            base_name = base_name[:30] + '…'
        questions.append(f"「{base_name}」の主な内容は何ですか？")

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
            f'<li style="font-size:14px;">{name}: <b>{cnt}</b> チャンク</li>'
            for name, cnt in source_counts.items()
        )
        st.markdown(
            f'<div style="border:1px solid #c3e6cb; border-radius:6px; padding:12px 16px; background:#f0fff4;">'
            f'<details>'
            f'<summary style="cursor:pointer; color:#28a745; font-weight:600; font-size:15px;">'
            f'ベクトル構築が完了しました！ 合計 {len(docs)} チャンク ／ {len(source_counts)} ファイル'
            f'</summary>'
            f'<ul style="margin-top:8px; line-height:1.8;">{file_items}</ul>'
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
            f'<li style="font-size:14px;">{name}: <b>{cnt}</b> チャンク</li>'
            for name, cnt in source_counts.items()
        )
        st.markdown(
            f'<div style="border:1px solid #c3e6cb; border-radius:6px; padding:12px 16px; background:#f0fff4;">'
            f'<details>'
            f'<summary style="cursor:pointer; color:#28a745; font-weight:600; font-size:15px;">'
            f'増分構築が完了しました！ 合計 {len(vector.document)} チャンク ／ {len(source_counts)} ファイル'
            f'</summary>'
            f'<ul style="margin-top:8px; line-height:1.8;">{file_items}</ul>'
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
    st.header("ナレッジベース")

    data_dir = './data'
    if os.path.isdir(data_dir):
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.md', '.txt', '.pdf'))])
        if files:
            for f in files:
                col_f, col_d = st.columns([5, 1])
                with col_f:
                    st.markdown(f"📄 {f}")
                with col_d:
                    if st.button("✕", key=f"sidebar_del_{f}", help=f"{f} を削除"):
                        os.remove(os.path.join(data_dir, f))
                        load_knowledge_base.clear()
                        st.rerun()
        else:
            st.caption("資料がありません")
    else:
        st.caption("資料ディレクトリが存在しません")

    st.divider()
    st.header("回答設定")

    st.selectbox(
        "LLMモデル",
        options=list(MODEL_OPTIONS.keys()),
        key="model_name"
    )

    st.selectbox(
        "回答の長さ（トークン数）",
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

    k = st.slider("検索数 Top-K", 1, 5, 3)

    show_context = st.checkbox("検索内容を表示", True)

    if st.button("会話をクリア"):
        st.session_state.history = []
        st.session_state.context_docs = []
        st.session_state.pending_question = None
        st.rerun()

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

col_left, col_right = st.columns([3, 1])

with col_left:
    if doc_summaries:
        file_list_html = ''.join(
            f'<div style="margin-top:4px;font-size:12px;color:rgba(255,255,255,0.9);">'
            f'📄 <b>{fname}</b> — {summary}</div>'
            for fname, summary in doc_summaries
        )
        st.markdown(
            f'<div style="background:linear-gradient(135deg, #3a7bd5 0%, #3a6073 100%);'
            f'padding:10px 16px;border-radius:10px;'
            f'min-height:80px;display:flex;flex-direction:column;justify-content:center;">'
            f'<span style="color:white;font-size:13px;font-weight:600;">'
            f'📚 ナレッジベース（{len(doc_summaries)}件のドキュメント）</span>'
            f'{file_list_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown(
        '<div style="border:2px dashed #aaa; border-radius:10px; padding:18px 8px; '
        'text-align:center; color:#999; background:#fafafa; font-size:12px; '
        'min-height:80px; display:flex; flex-direction:column; justify-content:center;">'
        '<div>Drag & Drop</div>'
        '<div style="color:#bbb;">.md .txt .pdf</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_main_{st.session_state.file_uploader_counter}",
        label_visibility="collapsed",
    )

    if uploaded:
        for uf in uploaded:
            filepath = os.path.join(data_dir, uf.name)
            with open(filepath, 'wb') as wf:
                wf.write(uf.getbuffer())
        st.session_state.file_uploader_counter += 1
        load_knowledge_base.clear()
        st.rerun()

loading_placeholder = st.empty()
loading_placeholder.markdown(
    '<div style="display:flex; justify-content:center; align-items:center; min-height:180px;">'
    '<div style="text-align:center; padding:36px 56px; border:1px solid #e0e0e0; '
    'border-radius:12px; background:#ffffff; box-shadow:0 2px 16px rgba(0,0,0,0.06);">'
    '<div style="font-size:18px; font-weight:600; color:#333;">'
    'ナレッジベースを読み込み中...</div>'
    '</div></div>',
    unsafe_allow_html=True,
)
vector, embedding = load_knowledge_base()
loading_placeholder.empty()

st.markdown('<hr style="margin:10px 0 6px 0;border:none;border-top:1px solid #e0e0e0;">', unsafe_allow_html=True)

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

sample_questions = get_sample_questions(doc_summaries)
st.markdown(
    '<p style="font-size:13px;color:#888;margin-top:10px;margin-bottom:6px;">質問例をクリックして試してみましょう</p>',
    unsafe_allow_html=True,
)
cols = st.columns(len(sample_questions))
for i, (col, q) in enumerate(zip(cols, sample_questions)):
    with col:
        if st.button(
            q, key=f"sample_q_{st.session_state.file_uploader_counter}_{i}", use_container_width=True,
            type="secondary",
        ):
            st.session_state.pending_question = q
            st.rerun()

question = st.chat_input("質問を入力してください...")

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
