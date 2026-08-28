"""
RAG Pipeline — Rebuilt with ChromaDB
FlyRank Content Intelligence Platform

Replaces the original DocArrayInMemorySearch scaffold with:
- Persistent ChromaDB vector store
- Auto URL fetching for latest SEO knowledge
- RetrievalQA chain with LangChain (when LLM is available)
- Graceful fallback to retrieval-only when LLM not configured
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.retriever import retrieve, retrieve_for_page, add_url_to_knowledge, get_collection
from Agents.model import llm, LLM_AVAILABLE

# ── Optional LangChain RetrievalQA setup ─────────────────────────────────────
_qa_chain = None

def get_qa_chain():
    """Build a LangChain RetrievalQA chain backed by ChromaDB if LLM is available."""
    global _qa_chain
    if _qa_chain is not None:
        return _qa_chain
    if not LLM_AVAILABLE:
        return None

    try:
        import chromadb
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain.chains import RetrievalQA
        from langchain_core.prompts import PromptTemplate
        from Agents.model import API_KEY, BASE_URL

        CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")

        embeddings = OpenAIEmbeddings(
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL if BASE_URL else None,
        )
        vectorstore = Chroma(
            collection_name="flyrank_seo_knowledge",
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        prompt_template = """You are a senior SEO strategist at FlyRank.
Use the following SEO knowledge to answer the question.
If you don't know, say so — do not make up information.

SEO Knowledge:
{context}

Question: {question}

Answer (be specific and actionable):"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            verbose=False,
        )
        print("[RAG] RetrievalQA chain ready.")
    except Exception as e:
        print(f"[RAG] Could not build RetrievalQA chain: {e}")
        _qa_chain = None

    return _qa_chain


def query_knowledge(question: str) -> str:
    """
    Query the SEO knowledge base with a natural language question.
    Uses LLM-powered RetrievalQA if available, otherwise returns raw chunks.
    """
    chain = get_qa_chain()
    if chain:
        try:
            return chain.run(question)
        except Exception as e:
            print(f"[RAG] Chain error: {e}")

    # Fallback: return raw retrieved chunks
    docs = retrieve(question, n_results=3)
    if not docs:
        return "No relevant knowledge found."
    return "\n\n".join([d["text"][:400] for d in docs])


def analyze_page_with_rag(page_data: dict) -> dict:
    """
    Full RAG analysis for a page:
    1. Auto-builds query from page metrics
    2. Retrieves relevant SEO knowledge
    3. If LLM available → generates specific recommendations
    4. Returns structured result
    """
    # Step 1: Auto-retrieve knowledge based on page metrics
    docs = retrieve_for_page(page_data, n_results=4)
    context = "\n\n".join([d["text"][:500] for d in docs]) if docs else "General SEO best practices apply."

    # Step 2: LLM-powered recommendation
    llm_recommendation = None
    if LLM_AVAILABLE and llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            sys_msg = SystemMessage(content=(
                "You are a senior SEO strategist at FlyRank. "
                "Give specific, actionable recommendations based on the page data and SEO knowledge provided."
            ))
            user_msg = HumanMessage(content=(
                f"Page data: {page_data}\n\n"
                f"Relevant SEO knowledge:\n{context}\n\n"
                "Provide:\n"
                "1. One-sentence refresh priority reason\n"
                "2. Suggested SEO title (max 60 chars)\n"
                "3. Suggested meta description (max 155 chars)\n"
                "4. Top 3 content improvements"
            ))
            response = llm.invoke([sys_msg, user_msg])
            llm_recommendation = response.content
        except Exception as e:
            llm_recommendation = f"LLM error: {e}"

    return {
        "retrieved_docs":    docs,
        "context_used":      context[:800],
        "llm_recommendation": llm_recommendation,
        "llm_available":     LLM_AVAILABLE,
    }


if __name__ == "__main__":
    # Initialize knowledge base (fetches URLs automatically)
    print("Initializing RAG knowledge base...")
    get_collection()

    # Test query
    result = analyze_page_with_rag({
        "ctr": 0.01,
        "impressions_90d": 800,
        "avg_position": 6.5,
        "content_age_days": 450,
        "score": 0.78,
        "reason_code": "DECAY",
    })
    print(f"\nRAG analysis complete. LLM used: {result['llm_available']}")
    print(f"Retrieved {len(result['retrieved_docs'])} docs.")
    if result["llm_recommendation"]:
        print(f"\nLLM Recommendation:\n{result['llm_recommendation']}")