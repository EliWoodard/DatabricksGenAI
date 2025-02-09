const slider = document.querySelector('.slider');

function activate(e) {
    const items = document.querySelectorAll('.item');
    if (e.target.matches('.next')) {
        slider.append(items[0]);
    } else if (e.target.matches('.prev')) {
        slider.prepend(items[items.length - 1]);
    }
}

function showModal(title, text = "") {
    const modal = document.getElementById("modal");
    const modalTitle = document.getElementById("modal-title");
    const modalText = document.getElementById("modal-text");

    // Set the title and text content
    modalTitle.textContent = title;

    // Only set text content if it's provided
    if (text) {
        modalText.innerHTML = text;  // Use innerHTML to allow formatted text
    }

    // Make the modal visible
    modal.classList.add("visible");
}


function closeModal() {
    const modal = document.getElementById("modal");
    const modalTitle = document.getElementById("modal-title");
    const modalText = document.getElementById("modal-text");

    // Hide the modal
    modal.classList.remove("visible");

    // Clear modal content
    modalTitle.textContent = "";
    modalText.innerHTML = "";
}

// Store the entire Python code as a multi-line string
const codeSections = [
    {
        title: "Web Scraper",
        code: `
# ===================================================
# Webscrapper 
# ===================================================
import requests
from bs4 import BeautifulSoup
import time

def get_internal_links(soup, base_url):
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/') or base_url in href:
            full_link = base_url + href if href.startswith('/') else href
            links.append(full_link)
    return links

def scrape_all_pages(base_url):
    visited = set()
    to_visit = [base_url]
    all_text_data = ""

    while to_visit:
        url = to_visit.pop(0)
        if url not in visited:
            visited.add(url)
            print(f"Scraping {url}...")

            # Scrape page content
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Collect text
                page_text = soup.get_text(separator=" ", strip=True)
                all_text_data += page_text + "\\n"

                # Find new links to visit
                internal_links = get_internal_links(soup, base_url)
                to_visit.extend([link for link in internal_links if link not in visited])
            except requests.RequestException as e:
                print(f"Failed to scrape {url}: {e}")

    return all_text_data

# Start scraping from the homepage
base_url = "https://smoothstack.com"
all_content = scrape_all_pages(base_url)
print(all_content)`
    },
    {
        title: "Storing Text",
        code: `
with open("smoothstack_text_data.txt", "w", encoding="utf-8") as file:
    file.write(all_content)`
    },
    {
        title: "Setting Database Path",
        code: `
%sql
use catalog gen_ai_morning;
use schema elijah_schema`
    },
    {
        title: "Building and Using our Agent",
        code: `
# ===================================================
# 1. Imports & Setup
# ===================================================
from pyspark.sql import SparkSession
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from databricks_langchain import ChatDatabricks

# Tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

# Agent
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import initialize_agent, AgentType, Tool

# Create or get existing Spark session
spark = SparkSession.builder.appName("GenAIChatbotApp").getOrCreate()

# ===================================================
# 2. Load Document & Chunk Data
# ===================================================
document_path = "/Workspace/Users/elijah.woodard@smoothstack.com/GenAIChatbotApp/smoothstack_text_data.txt"

# Load document
document = TextLoader(document_path).load()
if not document:
    print("Failed to load document.")
    exit()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

chunks = text_splitter.split_documents(documents=document)

print(f"Total chunks created: {len(chunks)}")
print("=== Sample Chunk ===")
print(chunks[0].page_content[:500], "...")

# ===================================================
# 3. Store Chunks in Spark DataFrame
# ===================================================
chunk_data = [(i, chunk.page_content) for i, chunk in enumerate(chunks)]
spark_df = spark.createDataFrame(chunk_data, ["chunk_id", "text"])

# Display sample from Spark DataFrame
display(spark_df.limit(5))

# ===================================================
# 4. Create Delta Table
# ===================================================
# Save as Delta table & Check if the table exists
table_name = "elijah_schema.GenAIChatbotAppChunks"
table_exists = spark.catalog.tableExists(table_name)

if not table_exists:
    # Save the DataFrame as a Delta table
    spark_df.write.format("delta").mode("overwrite").saveAsTable(table_name)

    # Optimize the table with ZORDER indexing
    spark.sql(f"""
        OPTIMIZE {table_name}
        ZORDER BY (chunk_id)
    """)
    print("=== Delta table created and optimized ===")
else:
    print("=== Delta table already exists ===")

# ===================================================
# 5. Implement Retrieval-Augmented Generation (RAG)
# ===================================================
# Embeddings and Vector Store setup
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = Chroma.from_documents(chunks, embedding)
retriever = vector_store.as_retriever()

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for answering questions. You have access to RAG Retrieval, Wikipedia, and DuckDuckGo search engine. Use the RAG for any query that is regarding smoothstack information. Use the other tools for anything that you don't know.
    If the answer isn't clear, acknowledge that you don't know. Limit your response to three concise sentences.
     {context}"""),
    ("human", "{input}")
])

# Create the RAG chain
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)
qa_chain = create_stuff_documents_chain(chat_model, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ===================================================
# 6. Define tools
# ===================================================
duckduckgo_tool = DuckDuckGoSearchRun()
wikipedia_tool = WikipediaAPIWrapper()

# Define the RAG chain as a tool
rag_tool = Tool(
    name="RAG Retrieval",
    description="Useful for retrieving relevant information from internal documents related to smoothstack.",
    func=lambda query: rag_chain.invoke({"input": query})['answer']
)

# Update tools list to include RAG
tools = [
    Tool(
        name="DuckDuckGo Search",
        description="Useful for searching the web when additional context or up-to-date information is needed.",
        func=duckduckgo_tool.run
    ),
    Tool(
        name="Wikipedia Search",
        description="Useful for retrieving information from Wikipedia.",
        func=wikipedia_tool.run
    ),
    rag_tool  # Add the RAG tool
]

# ===================================================
# 8. Initialize agent with tools
# ===================================================
agent_chain = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# ===================================================
# 7. Ask a Question (RAG Demo)
# ===================================================
print("Chat with your data")

chat_history = []

while True:
    question = input("What is your question? (type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    # Use the agent chain to get a response
    response = agent_chain.run(input=question, chat_history=chat_history)
    print(f"Answer: {response}")

    # Update chat history
    chat_history.append({"input": question, "response": response})`
    }
];

// This function loads the code into the modal
function loadAllCodeSections() {
    const modalText = document.getElementById("modal-text");

    // Clear any previous content
    modalText.innerHTML = "";

    // Loop through each code section and create a box
    codeSections.forEach(section => {
        // Create a container for each code block
        const codeContainer = document.createElement("div");
        codeContainer.className = "code-container";

        // Add title
        const titleElement = document.createElement("h2");
        titleElement.textContent = section.title;
        codeContainer.appendChild(titleElement);

        // Add code block
        const codeBlock = document.createElement("pre");
        codeBlock.innerHTML = `<code class="language-python">${section.code}</code>`;
        codeContainer.appendChild(codeBlock);

        // Append the container to the modal
        modalText.appendChild(codeContainer);
    });

    // Show the modal
    showModal("Code Highlights", "");

    // Re-run Highlight.js to apply syntax highlighting
    if (typeof hljs !== "undefined") {
        setTimeout(() => hljs.highlightAll(), 50);
    }
}

// highlightAll() upon DOM load for any static code blocks
document.addEventListener('DOMContentLoaded', () => {
    hljs.highlightAll();
});

document.addEventListener('click', activate, false);
