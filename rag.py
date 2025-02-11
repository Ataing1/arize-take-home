from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState
from typing_extensions import List, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
import argparse
import halo
from langgraph.prebuilt import ToolNode
from loader import ResearchLoader
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

loader = ResearchLoader()
spinner = halo.Halo(text="Loading documents...", spinner="dots")
spinner.start()
docs = loader.load_all()
spinner.succeed("Documents loaded successfully")

spinner.start("Indexing documents...")
_ = vector_store.add_documents(documents=docs)
spinner.succeed("Documents indexed successfully")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


@tool(response_format="content")
def retrieve(state: State):
    """Retrieve relevant documents from the vector store based on the input question.

    Args:
        state: The current state containing the question to search for

    Returns:
        dict: A dictionary containing the retrieved documents in the 'context' key
    """
    spinner.start("fetching documents...")
    k = 10
    retrieved_docs = vector_store.similarity_search(state["question"], k=k)
    if args.verbose:
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument {i+1}:")
            print(
                f"Source: {doc.metadata.get('title')} ({doc.metadata.get('section', 'unknown section')})"
            )
            print(f"Key concepts: {', '.join(doc.metadata.get('key_concepts', []))}")

    # Add title to document content for context
    for doc in retrieved_docs:
        if doc.metadata.get("title"):
            doc.page_content = (
                f"Title: {doc.metadata.get('title')}\n\nContent: {doc.page_content}"
            )
    spinner.succeed(f"fetched {k} documents")
    return {"context": retrieved_docs}


SYSTEM_TEMPLATE = """You are an expert research assistant specializing in analyzing academic papers and technical documents. 

When answering questions:
1. Focus on information explicitly stated in the provided context
2. Cite specific papers and their sections when providing information
3. If the context is insufficient, acknowledge the limitations
4. Structure responses to address key aspects systematically
5. Highlight any relevant technical concepts or methodologies
6. Note any gaps or ambiguities in the retrieved context

Context:
{context}

Remember to maintain academic rigor while keeping responses clear and concise.
"""


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. When referencing papers, include their titles."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


def main():
    parser = argparse.ArgumentParser(description="Research QA System")
    parser.add_argument(
        "--verbose", action="store_true", help="Print retrieved documents"
    )
    global args
    args = parser.parse_args()

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    config = {"configurable": {"thread_id": "abc123"}}
    memory = MemorySaver()

    spinner = halo.Halo(text="Initializing Research QA System...", spinner="dots")
    spinner.start()
    graph = graph_builder.compile(checkpointer=memory)
    spinner.succeed("Research QA System initialized!")

    print("\nWelcome to the Research QA System! Type 'quit' to exit.")

    while True:
        user_input = input("\nEnter your question: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        print("\nProcessing your question...\n")

        print(f"Human: {user_input}\n")

        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            # Only print AI's final response (non-tool messages)
            last_message = step["messages"][-1]
            if last_message.type == "ai" and not last_message.tool_calls:
                print(f"\nAssistant: {last_message.content}\n")


if __name__ == "__main__":
    main()
