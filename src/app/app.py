"""
Interface Streamlit pour le chatbot RAG.

Ce fichier réutilise directement le système de retrieval hybride
(BM25 + vecteurs) défini dans `ragsystem.core.retriever.ChromaRetriever`,
le même que celui utilisé dans le script d'évaluation.

La génération de réponse passe par la classe `GroqLLM` (wrapper Groq avec
chaîne de modèles de secours), et non plus par un appel direct au SDK Groq.

Lancement (depuis la racine du dépôt) :
    streamlit run src/app/app.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import streamlit as st

from ragsystem.core.retriever import ChromaRetriever

# ---------------------------------------------------------------------------
# Adaptez ce chemin d'import à l'emplacement réel de votre classe GroqLLM
# dans votre projet (ex : "from llm.groq_llm import GroqLLM").
# ---------------------------------------------------------------------------
try:
    from ragsystem.utils.llm_connector import GroqLLM

    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


# =============================================================================
# CONFIGURATION PAR DÉFAUT (identique à eval.py)
# =============================================================================

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

DEFAULT_TOP_K = 5
DEFAULT_BM25_TOP_K = 5
DEFAULT_VECTOR_TOP_K = 5
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5

# Configuration de GroqLLM
GROQ_CONFIG_PATH = "./secrets/config.yaml"
GROQ_PRIMARY_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024

PAGE_TITLE = "RAG Assistant"
PAGE_ICON = "🧠"

# =============================================================================
# CHATBOT PROMPT
# =============================================================================
# This prompt internally receives the passages retrieved by the retriever
# (RAG), but the model must NEVER reveal that it relies on "excerpts",
# "chunks", or a "context": it must answer as if it directly knew the
# information, in a fluent and natural way. The user should only see a
# clean answer — the sources remain visible separately in the interface's
# "sources" panel, not in the text of the response itself.

CHATBOT_SYSTEM_PROMPT = """You are an expert conversational assistant who helps users \
by answering their questions clearly, naturally, and directly.

You are internally provided with reference information to help you answer. \
You must use it as your knowledge base, but:

- NEVER mention the words "excerpt", "chunk", "context", "source number", \
"retrieved document", "RAG", or any other reference to the search mechanism.
- Never say things like "based on the provided context" or "according to the excerpts": \
answer directly, as if you already knew the information.
- Write a fluent, well-structured response (complete sentences, bullet points if useful), \
without tags or source numbering in the body of the text.
- If the necessary information is not available in what you were given, simply say \
that you don't have that information, without explaining why or mentioning a \
document database.
- Never make up facts that are not supported by the provided information.
- Adapt the length of your response to the question: stay concise for a simple question, \
elaborate if the question requires it.

Always respond in the language used by the user."""


# =============================================================================
# STYLE — CSS personnalisé pour une interface bleue claire et sophistiquée
# =============================================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --blue-50:  #eef6ff;
        --blue-100: #dcecff;
        --blue-200: #bfe0ff;
        --blue-400: #4da3ff;
        --blue-500: #2f8bf0;
        --blue-600: #1f6fd1;
        --ink-900:  #12233b;
        --ink-600:  #4a5b73;
        --ink-400:  #7a8ba3;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 16px !important;
        color: var(--ink-900);
    }

    /* Fond général — bleu très clair, sophistiqué */
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, var(--blue-100) 0%, transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, var(--blue-50) 0%, transparent 55%),
            linear-gradient(180deg, #ffffff 0%, #f5faff 100%);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, var(--blue-600), var(--blue-400));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .hero-subtitle {
        color: var(--ink-400);
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1.3rem;
        letter-spacing: 0.01em;
    }

    .stChatMessage {
        background: #ffffff;
        border: 1px solid var(--blue-100);
        box-shadow: 0 2px 10px rgba(47, 139, 240, 0.06);
        border-radius: 14px;
        padding: 0.6rem 0.5rem;
    }
    .stChatMessage p, .stChatMessage li {
        font-size: 1.1rem !important;
        line-height: 1.65;
        color: var(--ink-900);
    }

    .source-card {
        background: #ffffff;
        border: 1px solid var(--blue-100);
        border-left: 3px solid var(--blue-400);
        border-radius: 10px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 6px rgba(47, 139, 240, 0.05);
    }
    .source-card-title {
        font-weight: 700;
        color: var(--ink-900);
        font-size: 0.95rem;
    }
    .source-card-meta {
        color: var(--ink-400);
        font-size: 0.85rem;
        margin-top: 0.1rem;
    }
    .score-pill {
        display: inline-block;
        padding: 0.1rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-right: 0.3rem;
        margin-top: 0.3rem;
    }
    .pill-hybrid { background: var(--blue-100); color: var(--blue-600); }
    .pill-bm25   { background: #e6faf5; color: #0e9f83; }
    .pill-vector { background: #eef1ff; color: #5b5bd6; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, var(--blue-50) 100%);
        border-right: 1px solid var(--blue-100);
    }
    section[data-testid="stSidebar"] * {
        font-size: 0.98rem !important;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 1.15rem !important;
        color: var(--ink-900);
        font-weight: 800;
    }

    .stTextInput input, .stSlider {
        font-size: 0.98rem !important;
    }
    .stButton button {
        border-radius: 10px !important;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        border: 1px solid var(--blue-200) !important;
        background: #ffffff !important;
        color: var(--blue-600) !important;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        background: var(--blue-50) !important;
        border-color: var(--blue-400) !important;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .status-on  { background: #e6faf5; color: #0e9f83; }
    .status-off { background: #fff6e0; color: #b9820b; }

    details {
        border-radius: 12px !important;
        border: 1px solid var(--blue-100) !important;
        background: #ffffff !important;
    }
    summary {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: var(--blue-600) !important;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 1.08rem !important;
    }
</style>
"""


# =============================================================================
# UTILITAIRES
# =============================================================================

@dataclass
class RagAnswer:
    text: str
    sources: list[dict[str, Any]]
    latency: float
    error: str | None = None


def format_score(score: float | None) -> str:
    if score is None:
        return "—"
    return f"{score:.3f}"


@st.cache_resource(show_spinner=False)
def get_retriever(
    chroma_dir: str,
    collection_name: str,
    bm25_weight: float,
    vector_weight: float,
) -> ChromaRetriever:
    """Instancie le retriever une seule fois et le met en cache.

    Les top_k ne font pas partie de la clé de cache : ils sont passés à
    chaque requête, ce qui évite de recharger le modèle d'embedding et de
    reconstruire l'index BM25 à chaque déplacement d'un slider.
    """

    return ChromaRetriever(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )


@st.cache_resource(show_spinner=False)
def get_llm(
    config_path: str,
    primary_model: str,
    fallback_models: tuple[str, ...],
):
    """
    Instancie GroqLLM une seule fois et le met en cache.

    Retourne (instance, erreur). En cas d'échec (fichier de config
    introuvable, clé manquante, etc.), l'instance est None et l'erreur
    contient le message d'exception, ce qui permet à l'app de basculer
    proprement en mode extractif plutôt que de planter.
    """

    if not _GROQ_AVAILABLE:
        return None, "Le module GroqLLM n'a pas pu être importé."

    try:
        llm = GroqLLM(
            config_path=config_path,
            primary_model=primary_model,
            fallback_models=list(fallback_models),
        )
        return llm, None
    except Exception as e:
        return None, str(e)


def build_context(retrieved: list[Any], max_chars: int = 6000) -> str:
    """Concatène le texte des chunks retrouvés en un contexte continu.

    Aucun repère de type "[Source N]" n'est inséré ici : on ne veut pas
    que le modèle cite des numéros de source dans sa réponse finale.
    Le mapping vers les vraies sources reste géré séparément par
    `answer_question`, pour l'affichage dans le panneau "sources" de l'UI.
    """

    parts = []
    used = 0

    for result in retrieved:
        chunk_text = getattr(result, "text", None) or getattr(
            result, "document", None
        ) or ""

        if not chunk_text:
            continue

        block = chunk_text.strip() + "\n"

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)


def generate_answer(
    question: str,
    context: str,
    llm,
) -> tuple[str | None, str | None]:
    """
    Génère une réponse naturelle via GroqLLM, sans exposer le mécanisme RAG.

    Retourne (réponse, erreur). Si `llm` est None (pas de LLM configuré),
    retourne (None, None) : c'est un signal pour basculer en mode extractif.
    Si le LLM est configuré mais que tous les modèles échouent, retourne
    (None, message_erreur).
    """

    if llm is None:
        return None, None

    # Le contexte est injecté avant la question, dans un seul prompt,
    # car GroqLLM.generate() ne distingue pas les rôles system/user.
    full_prompt = (
        f"{CHATBOT_SYSTEM_PROMPT}\n\n"
        f"--- Informations de référence (usage interne uniquement) ---\n"
        f"{context}\n"
        f"--- Fin des informations de référence ---\n\n"
        f"Question : {question}\n"
        f"Réponse :"
    )

    try:
        answer = llm.generate(
            prompt=full_prompt,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return answer, None
    except RuntimeError as e:
        return None, str(e)


def answer_question(
    retriever: ChromaRetriever,
    llm,
    question: str,
    top_k: int,
    bm25_top_k: int,
    vector_top_k: int,
) -> RagAnswer:
    start = time.time()

    retrieved = retriever.hybrid_search(
        query=question,
        top_k=top_k,
        bm25_top_k=bm25_top_k,
        vector_top_k=vector_top_k,
    )

    context = build_context(retrieved)
    llm_answer, llm_error = generate_answer(question, context, llm)

    if llm_answer is None:
        # Mode extractif : pas de LLM disponible ou génération échouée.
        if not retrieved:
            llm_answer = "Aucun document pertinent n'a été trouvé pour cette question."
        else:
            snippets = []
            for i, result in enumerate(retrieved, start=1):
                text = getattr(result, "text", None) or getattr(
                    result, "document", None
                ) or "(contenu indisponible)"
                snippets.append(f"**Extrait {i}** — {text[:400].strip()}...")
            llm_answer = (
                "🔎 *Mode extractif (génération LLM indisponible)* :\n\n"
                + "\n\n".join(snippets)
            )

    sources = []
    for result in retrieved:
        sources.append(
            {
                "id": result.id,
                "source": result.metadata.get("source"),
                "headings": result.metadata.get("headings"),
                "chunk_index": result.metadata.get("chunk_index"),
                "distance": result.distance,
                "bm25_score": result.bm25_score,
                "vector_score": result.vector_score,
                "hybrid_score": result.hybrid_score,
                "retrieval_sources": result.retrieval_sources,
            }
        )

    latency = time.time() - start

    return RagAnswer(
        text=llm_answer,
        sources=sources,
        latency=latency,
        error=llm_error,
    )


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Sidebar — configuration
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        chroma_dir = st.text_input("Répertoire Chroma", value=CHROMA_DIR)
        collection_name = st.text_input(
            "Nom de la collection", value=COLLECTION_NAME
        )

        st.markdown("---")
        st.markdown("**Paramètres de recherche**")

        top_k = st.slider("Top K final", 1, 20, DEFAULT_TOP_K)
        bm25_top_k = st.slider("Top K BM25", 1, 20, DEFAULT_BM25_TOP_K)
        vector_top_k = st.slider("Top K Vecteurs", 1, 20, DEFAULT_VECTOR_TOP_K)

        st.markdown("---")
        st.markdown("**Modèle Groq**")

        config_path = st.text_input(
            "Fichier de configuration (yaml)",
            value=GROQ_CONFIG_PATH,
            help="Fichier contenant la clé GROQ_API_KEY, lu par GroqLLM.",
        )
        primary_model = st.text_input(
            "Modèle principal", value=GROQ_PRIMARY_MODEL
        )
        fallback_input = st.text_input(
            "Modèles de secours (séparés par des virgules)",
            value=", ".join(GROQ_FALLBACK_MODELS),
        )
        fallback_models = tuple(
            m.strip() for m in fallback_input.split(",") if m.strip()
        )

        llm, llm_error = get_llm(config_path, primary_model, fallback_models)
        llm_ready = llm is not None

        badge_class = "status-on" if llm_ready else "status-off"
        badge_text = (
            "Génération LLM active ✅" if llm_ready else "Mode extractif (pas de LLM) ⚠️"
        )
        st.markdown(
            f'<span class="status-badge {badge_class}">{badge_text}</span>',
            unsafe_allow_html=True,
        )

        if not llm_ready and llm_error:
            st.caption(f"Détail : {llm_error}")

        st.markdown("---")
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ------------------------------------------------------------------
    # En-tête
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="hero-title">🧠 RAG Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Recherche hybride BM25 + Vecteurs · '
        'Réponses ancrées sur vos documents</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Initialisation du retriever
    # ------------------------------------------------------------------
    try:
        retriever = get_retriever(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            bm25_weight=DEFAULT_BM25_WEIGHT,
            vector_weight=DEFAULT_VECTOR_WEIGHT,
        )
    except Exception as e:
        st.error(f"❌ Impossible d'initialiser le retriever : {e}")
        st.stop()

    # ------------------------------------------------------------------
    # Historique de conversation
    # ------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(
                    f"📚 {len(message['sources'])} source(s) utilisée(s) "
                    f"· {message.get('latency', 0):.2f}s"
                ):
                    render_sources(message["sources"])

    # ------------------------------------------------------------------
    # Entrée utilisateur
    # ------------------------------------------------------------------
    question = st.chat_input("Posez votre question sur vos documents…")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération de la réponse…"):
                result = answer_question(
                    retriever=retriever,
                    llm=llm,
                    question=question,
                    top_k=top_k,
                    bm25_top_k=bm25_top_k,
                    vector_top_k=vector_top_k,
                )

            if result.error:
                st.warning(
                    f"⚠️ La génération LLM a échoué, réponse en mode extractif "
                    f"— {result.error}"
                )

            st.markdown(result.text)

            if result.sources:
                with st.expander(
                    f"📚 {len(result.sources)} source(s) utilisée(s) "
                    f"· {result.latency:.2f}s"
                ):
                    render_sources(result.sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.text,
                "sources": result.sources,
                "latency": result.latency,
            }
        )


def render_sources(sources: list[dict[str, Any]]) -> None:
    for i, source in enumerate(sources, start=1):
        headings = source.get("headings") or ""
        source_name = source.get("source") or "inconnu"
        chunk_index = source.get("chunk_index")

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-card-title">#{i} — {source_name}</div>
                <div class="source-card-meta">
                    {headings if headings else ""}
                    {f" · chunk #{chunk_index}" if chunk_index is not None else ""}
                </div>
                <div>
                    <span class="score-pill pill-hybrid">hybrid {format_score(source.get('hybrid_score'))}</span>
                    <span class="score-pill pill-bm25">bm25 {format_score(source.get('bm25_score'))}</span>
                    <span class="score-pill pill-vector">vector {format_score(source.get('vector_score'))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
