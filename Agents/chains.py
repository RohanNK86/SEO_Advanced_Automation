"""
LangChain Chains Configuration
FlyRank Content Intelligence Platform

Fixed version: correct imports and memory setup.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Agents.model import llm, LLM_AVAILABLE

if LLM_AVAILABLE:
    from langchain.chains import LLMChain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain.chains import ConversationChain

    # Generic SEO analysis prompt chain
    seo_prompt = ChatPromptTemplate.from_template(
        "You are a senior SEO strategist at FlyRank.\n"
        "Context from knowledge base:\n{context}\n\n"
        "Page data:\n{page_data}\n\n"
        "Provide: 1) Refresh priority reason, 2) Title suggestion, "
        "3) Meta description, 4) Top 3 content improvements."
    )

    seo_chain = LLMChain(llm=llm, prompt=seo_prompt) if llm else None

    # Conversation chain with memory (for interactive review)
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500) if llm else None
    conversation = ConversationChain(
        memory=memory,
        llm=llm,
        verbose=False
    ) if llm else None

else:
    seo_chain   = None
    conversation = None
    print("[chains] LLM not configured — chain disabled")