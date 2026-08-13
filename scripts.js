// Variables
const slider = document.querySelector(".slider");
const modal = document.getElementById("modal");
const modalTitle = document.getElementById("modal-title");
const modalText = document.getElementById("modal-text");
const modalCloseButton = document.querySelector(".modal-content .close");
const modalDetails = {
    certification: {
        title: "Certification",
        html: `This project directly supports Databricks certification requirements—particularly for the Generative AI Engineer Associate—by demonstrating real-world use of Spark, Delta tables, and scalable machine learning workflows. We developed an end-to-end pipeline that ingests website data through a custom Python scraper, transforms it with Spark, and stores chunked documents in a Delta table for efficient query and retrieval. The project showcases the ability to work with advanced libraries (like LangChain and Databricks ML tools) for Retrieval-Augmented Generation (RAG), aligning with certification objectives for building enterprise-grade data and AI solutions. By integrating a vector store, LLM-based query chains, and additional search tools (e.g., DuckDuckGo, Wikipedia), the solution covers key competencies such as data engineering, ML orchestration, and multi-source retrieval—essential skills for any Databricks certification involving Gen AI.`
    },
    "project-overview": {
        title: "Project Overview",
        html: `The chatbot application leverages Databricks, LangChain, and external Python libraries to perform Retrieval-Augmented Generation (RAG). At a high level, the flow begins by scraping web data (e.g., corporate or public websites) with Python’s requests and BeautifulSoup, then chunking text into manageable segments using LangChain’s text splitters. These chunks are embedded, indexed in a vector store, and stored in a Delta table for scalable retrieval. Using Databricks notebooks and Spark, the pipeline ensures quick data processing and integration. When users query the chatbot, relevant chunks are retrieved, combined, and passed to a large language model—leveraging custom prompt templates and chain logic—to generate context-aware, concise responses. This end-to-end approach demonstrates how structured and unstructured data can be integrated in a sophisticated yet maintainable AI solution.`
    },
    tools: {
        title: "Tools",
        html: `A diverse range of tools and libraries power this project:<br><br>
            1. <strong>LangChain</strong>: Provides modular components (e.g., document loaders, text splitters, RAG chain) for building advanced conversational applications.<br>
            2. <strong>Databricks</strong>: Manages data engineering tasks such as chunk storage in Delta tables and parallel processing via Spark, ensuring scalability and reliability.<br>
            3. <strong>OpenAI &amp; ChatDatabricks</strong>: Offers embeddings for semantic search and a large language model to respond intelligently to user queries.<br>
            4. <strong>BeautifulSoup &amp; Requests</strong>: Used for web scraping, retrieving raw HTML from target URLs, then parsing the DOM to extract relevant text.<br>
            5. <strong>Chroma</strong>: Serves as the vector store for embedding-based retrieval, enabling accurate, context-driven responses.<br>
            6. <strong>DuckDuckGo &amp; Wikipedia Tools</strong>: Integrates external knowledge sources, ensuring the chatbot can handle queries beyond the core dataset.<br><br>
            By carefully combining these tools, the project handles data ingestion, transformation, vector retrieval, and conversation management in one seamless pipeline.`
    },
    "use-cases": {
        title: "Use Cases",
        html: `This solution addresses multiple real-world scenarios:<br><br>
            <ul>
                <li><strong>Internal Knowledge Base</strong>:&nbsp;Automate employee queries about products, policies, and processes by scraping internal documentation and storing it for quick retrieval.</li>
                <li><strong>Customer Support</strong>:&nbsp;Provide context-aware support responses by integrating relevant articles from internal docs, Wikipedia, or web search tools.</li>
                <li><strong>Research &amp; Insights</strong>:&nbsp;Quickly search large text corpora—like scholarly articles or compliance documentation—using advanced chunking, embedding, and retrieval.</li>
                <li><strong>External Data Integration</strong>:&nbsp;Pull information from the web (e.g., DuckDuckGo or Wikipedia) to enrich local data, giving users more comprehensive and up-to-date answers.</li>
            </ul>
            <br>Whether for enterprise Q&amp;A, real-time support, or data-driven research, the platform’s combination of RAG, vector storage, and Spark-based orchestration is designed to handle large-scale, unstructured text with ease.`
    }
};
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

// Functions
function activate(e) {
    const control = e.target.closest(".prev, .next");

    if (!control || !slider) return;

    const items = slider.querySelectorAll(".item");
    closeModal();

    if (control.classList.contains("next")) {
        slider.append(items[0]);
    } else {
        slider.prepend(items[items.length - 1]);
    }
}

function closeModal() {
    modal.classList.remove("visible");
    modalTitle.textContent = "";
    modalText.replaceChildren();
}

function createCodeSection(section) {
    const codeContainer = document.createElement("section");
    const titleElement = document.createElement("h2");
    const codeBlock = document.createElement("pre");
    const codeElement = document.createElement("code");

    codeContainer.className = "code-container";
    titleElement.textContent = section.title;
    codeElement.className = "language-python";
    codeElement.textContent = section.code.trim();

    codeBlock.appendChild(codeElement);
    codeContainer.append(titleElement, codeBlock);

    if (typeof hljs !== "undefined") {
        hljs.highlightElement(codeElement);
    }

    return codeContainer;
}

function handleSlideAction(event) {
    const button = event.target.closest("[data-modal-id]");

    if (!button || !slider?.contains(button)) {
        return;
    }

    const modalId = button.dataset.modalId;

    if (modalId === "code-highlights") {
        loadAllCodeSections();
        return;
    }

    const detail = modalDetails[modalId];

    if (detail) {
        showModal(detail.title, detail.html);
    }
}

function loadAllCodeSections() {
    const fragment = document.createDocumentFragment();

    codeSections.forEach((section) => {
        fragment.appendChild(createCodeSection(section));
    });

    showModal("Code Highlights");
    modalText.appendChild(fragment);
}

function showModal(title, text = "") {
    modalTitle.textContent = title;
    modalText.innerHTML = text;
    modal.classList.add("visible");
}

// Event listeners
document.addEventListener("DOMContentLoaded", () => {
    if (typeof hljs !== "undefined") {
        hljs.highlightAll();
    }
});

document.querySelector(".nav")?.addEventListener("click", activate);
slider?.addEventListener("click", handleSlideAction);
modalCloseButton?.addEventListener("click", closeModal);

modal?.addEventListener("click", (event) => {
    if (event.target === modal) {
        closeModal();
    }
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && modal?.classList.contains("visible")) {
        closeModal();
    }
});
