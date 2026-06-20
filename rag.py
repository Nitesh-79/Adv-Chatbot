from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain(llm):

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using the provided context.

    Context:
    {context}

    Question:
    {question}
    """)

    chain = prompt | llm | StrOutputParser()

    return chain