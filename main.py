import sys
import os
from dotenv import load_dotenv
from rich import print
# Standard LangChain LCEL components: Prompt Templates & Output Parsers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Google Gemini provider integration
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    # Load environment variables from .env
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        print("Warning: Neither LLM_API_KEY nor LLM_API_KEY found in environment or .env file.")
        print("Please set your API key in a .env file or environment variable.\n")

    # Get user question from CLI arguments or prompt interactively
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask a question: ").strip()

    if not question:
        print("No question provided. Exiting.")
        return

    # 1. Standard LangChain Prompt Template (system instructions + variable placeholder)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, accurate, and concise AI assistant."),
        ("human", "{question}"),
    ])

    # 2. Model Initialization (gemini-3.1-flash-lite)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
    )

    # 3. Output Parser: parses the AIMessage object directly into a plain string
    parser = StrOutputParser()

    # 4. Standard LangChain Expression Language (LCEL) Chain: Prompt -> LLM -> Parser
    chain = prompt | llm | parser

    print("\nSending question to Gemini via LangChain chain...")
    response = chain.invoke({"question": question})

    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()



#Testig for a comment #testting for test
#just testing for a comment