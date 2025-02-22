# Gemini-Powered Search Agent

This project implements an AI-powered search agent using Gemini, LangGraph, and a set of tools to answer user queries by leveraging external information sources. The agent is designed to understand user requests, determine if external information is needed, select and use appropriate tools to gather that information, and then provide a well-informed, cited response. A Streamlit application is included to provide an interactive user interface for this agent.

## Project Overview

The goal of this project is to demonstrate how to build a sophisticated search agent that can:

  - **Understand User Queries:** Process natural language queries from users.
  - **Tool Utilization:** Decide when and which external tools are necessary to answer a query.
  - **Information Retrieval:** Use tools to search Wikipedia, DuckDuckGo, and fetch web page content.
  - **Cited Responses:**  Provide answers based on information gathered from external tools, ensuring sources can be cited.
  - **Iterative Search:** Employ multiple tools and refine searches if initial tool responses are insufficient.
  - **Interactive Interface:** Offer a user-friendly interface through a Streamlit application to interact with the agent and view conversation history.

## Key Components and Files

### 1\. `agent.py` - The Intelligent Agent Core

This file defines the core logic of the search agent using LangGraph to orchestrate interactions with the Gemini model and available tools.

**Key Concepts:**

  - **`SearchAgent` Class:** This class encapsulates the agent's behavior, including:

      - **Initialization (`__init__`)**:
          - Loads the Gemini model (`gemini-2.0-flash-exp` by default).
          - Configures the model with a system instruction that defines the agent's role and constraints (be helpful, use tools, cite sources, exhaust search before saying "I don't know").
          - Sets up available tools and a mapping for easy access.
          - Builds the LangGraph graph that defines the agent's workflow.
      - **`call_llm(state: AgentState)`**: This function sends the current conversation history (`state.messages`) to the Gemini model and receives a response. It also handles retry logic for robust API calls.
      - **`use_tool(state: AgentState)`**:  When the Gemini model decides to use a tool (indicated by a `function_call` in the response), this function:
          - Extracts the tool name and arguments from the model's response.
          - Retrieves the corresponding tool function from `self.tool_mapping`.
          - Executes the tool with the provided arguments.
          - Formats the tool's response to be fed back into the conversation history.
      - **`should_we_stop(state: AgentState) -> str`**: This crucial function determines the agent's next step based on the model's response:
          - If the latest model response contains a `function_call`, it means the agent should use a tool.  It returns `"use_tool"` to direct the LangGraph flow to the `use_tool` node.
          - Otherwise, it means the model has produced a final answer. It returns `END` to terminate the LangGraph flow.
      - **`build_agent()`**: This function constructs the LangGraph state graph:
          - **Nodes**:
              - `"call_llm"`: Executes the `call_llm` function to get a response from the Gemini model.
              - `"use_tool"`: Executes the `use_tool` function to call a selected tool and process its response.
          - **Edges**:
              - `START -> "call_llm"`: The agent starts by calling the language model.
              - `"call_llm" -> "should_we_stop"`: After getting a model response, the agent decides whether to use a tool or stop.
              - `"use_tool" -> "call_llm"`: After using a tool and getting a response, the agent goes back to the language model to process the tool's output.
          - **Conditional Edges**:
              - `"call_llm" -> "should_we_stop"`:  Based on the output of `should_we_stop`, the flow goes to either `"use_tool"` or `END`.
          - **Compilation**: The `builder.compile()` method finalizes the graph for execution.
      - **`invoke(user_query: str)`**: This is the main function to interact with the agent. It:
          - Initializes the `AgentState` with the user's query as the first message.
          - Invokes the LangGraph graph (`self.graph.invoke(initial_state)`) to start the agent's workflow.
          - Extracts and returns the final text response from the agent's output.

  - **`AgentState` Class:**  A Pydantic `BaseModel` that defines the state of the agent's conversation. Currently, it only holds `messages`, which is a list accumulating message dictionaries throughout the conversation. `Annotated[list, add]` ensures that new messages are appended to the list using the `add` operator.

### 2\. `tools.py` - External Tools Definition

This file defines the tools that the agent can use to gather information from external sources.

**Key Concepts:**

  - **Tool Functions:**

      - **`search_wikipedia(search_query: str) -> SearchResponse`**:
          - Searches Wikipedia for pages related to the `search_query`.
          - Uses the `wikipedia` library to perform the search.
          - Returns a `SearchResponse` Pydantic model containing a list of `PageSummary` objects (title, summary, URL) for the top search results.
      - **`get_wikipedia_page(page_title: str, max_text_size: int = 16_000) -> FullPage`**:
          - Retrieves the full content of a Wikipedia page given its `page_title`.
          - Uses the `wikipedia` library to fetch the page and `strip_tags` to remove HTML markup, extracting plain text content.
          - Returns a `FullPage` Pydantic model containing the page title, URL, and content (truncated to `max_text_size`).
      - **`search_duck_duck_go(search_query: str) -> SearchResponse`**:
          - Searches DuckDuckGo for web pages related to the `search_query`.
          - Uses the `duckduckgo_search` library to perform the search.
          - Returns a `SearchResponse` Pydantic model similar to `search_wikipedia`, containing `PageSummary` objects for search results.
      - **`get_page_content(page_title: str, page_url: str) -> FullPage`**:
          - Fetches the content of an arbitrary web page given its `page_url`.
          - Uses the `requests` library to get the HTML content and `strip_tags` to extract plain text.
          - Returns a `FullPage` Pydantic model containing the page title, URL, and content.

  - **Pydantic Models:**  `tools.py` defines several Pydantic models (`PageSummary`, `SearchResponse`, `FullPage`, `ErrorResponse`) to structure the data returned by the tool functions. This ensures type safety and clear data contracts.

  - **`@catch_exceptions` Decorator:** This decorator is applied to each tool function to handle potential exceptions gracefully. If a tool function fails (e.g., network error, Wikipedia page not found), it catches the exception, logs a warning, and returns an `ErrorResponse` model, preventing the agent from crashing and allowing it to handle errors.

  - **`@lru_cache` for Wikipedia Page Retrieval:**  The `wikipedia.page` function is decorated with `@lru_cache` to cache results and avoid redundant API calls for the same Wikipedia page title, improving efficiency.

### 3\. `app.py` - Streamlit User Interface

This file builds a Streamlit application to provide a user-friendly interface for interacting with the search agent.

**Key Features:**

  - **Streamlit UI:** Creates a simple web application with:

      - A title: "AI-Powered Search Agent".
      - A text input field for users to enter their queries.
      - A display area to show the conversation history.
      - A spinner to indicate when the agent is searching.
      - Output display to show the agent's responses.

  - **Agent Initialization:**  Instantiates the `SearchAgent` with the defined tools when the application starts.

  - **Conversation History Management:**

      - **Session State (`st.session_state`)**: Uses Streamlit's session state to maintain the conversation history during a user session.
      - **`initialize_session_state()`**: Initializes `st.session_state['chat_history']` as an empty list if it doesn't exist.
      - **`save_conversation(filename="conversation_history.json")`**: Saves the current `chat_history` to a JSON file (`conversation_history.json`) to persist conversations between sessions (simulating a simple database).
      - **`load_conversation(filename="conversation_history.json")`**: Loads conversation history from the JSON file when the application starts.

  - **Query Handling:**

      - When a user enters a query and presses Enter:
          - Displays a "Searching..." spinner.
          - Calls `agent.invoke(query)` to get the agent's response.
          - Displays the agent's response in the Streamlit app.
          - Appends the user query and agent response to `st.session_state['chat_history']`.
          - Saves the updated conversation history.
          - Handles potential exceptions during the agent invocation and displays error messages in the UI.

  - **Conversation History Display:** Displays the loaded and ongoing conversation history in the Streamlit app, showing user queries and agent responses with separators.

## Tools & Technologies Used

  - **Gemini API (google.generativeai):**  Google's Gemini model is the core language model powering the agent.
  - **LangGraph (langgraph.graph):**  Framework for building conversational agents with graph-based routing.
  - **Streamlit (streamlit):**  Python library for creating interactive web applications for data science and machine learning.
  - **DuckDuckGo Search API (duckduckgo\_search):** Python library to interface with the DuckDuckGo search engine.
  - **Wikipedia API (wikipedia):** Python library to access and parse Wikipedia content.
  - **Pydantic (pydantic):**  Data validation and settings management using Python type annotations, used for defining data models for tools and agent state.
  - **dotenv (dotenv):**  Loads environment variables from a `.env` file (used to securely manage the Gemini API key).
  - **strip-tags (strip\_tags):**  Simple library to remove HTML tags from text content.
  - **requests (requests):**  Python library for making HTTP requests (used to fetch web page content).
  - **operator (operator.add):** Used for list concatenation in `AgentState` using `Annotated`.
  - **typing (typing):** For type hinting, improving code readability and maintainability.
  - **functools (functools.lru\_cache, functools.wraps):** For caching results of functions and creating decorators.
  - **logger (logger):**  A simple logger module (assumed to be in the project, although not explicitly shown in the provided code snippets) for logging events and debugging.

## Setup and Run Instructions

1.  **Prerequisites:**

      - **Python 3.8 or higher:** Ensure you have Python installed on your system.
      - **pip:** Python package installer (usually comes with Python).
      - **Gemini API Key:** You need a Google Gemini API key. Obtain one from [Google AI Studio](https://www.google.com/url?sa=E&source=gmail&q=https://makersuite.google.com/).
      - **Create a `.env` file:** In the project directory, create a file named `.env` and add your Gemini API key:
        ```
        GEMINI_API_KEY=YOUR_GEMINI_API_KEY
        ```

2.  **Installation:**

      - **Clone the repository (if applicable) or create the project files:** Ensure you have `agent.py`, `tools.py`, `app.py`, and `logger.py` (if available) in the same directory.
      - **Navigate to the project directory in your terminal.**
      - **Create a `requirements.txt` file:**  Run the following command to automatically generate `requirements.txt` based on your project dependencies (make sure you have run the code at least once to install dependencies):
        ```bash
        pip freeze > requirements.txt
        ```
        Alternatively, create `requirements.txt` manually with the following content:
        ```txt
        duckduckgo-search
        pydantic
        strip-tags
        wikipedia
        google-generativeai
        google-api-core
        langgraph
        streamlit
        python-dotenv
        requests
        ```
      - **Install dependencies:** Run the following command to install all required Python libraries:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Run the Streamlit Application:**

      - In your terminal, navigate to the project directory.
      - Run the Streamlit app using the command:
        ```bash
        streamlit run app.py
        ```
      - Streamlit will launch the application in your web browser (usually at `http://localhost:8501`).

## How to Use the Streamlit Application

1.  Open the Streamlit application in your browser after running `streamlit run app.py`.
2.  You will see the "AI-Powered Search Agent" title and a text input field labeled "Enter your query:".
3.  Type your search query into the text input field. For example, "What are the main benefits of Stevia?".
4.  Press Enter or click outside the input field.
5.  The application will display a "Searching..." spinner while the agent processes your query.
6.  Once the agent finishes searching and generates a response, the "Results:" section will appear, displaying the agent's answer. If the agent was able to cite sources, the answer should be based on information retrieved from the tools.
7.  The conversation history (user queries and agent responses) will be displayed below the title.  This history is saved and loaded from `conversation_history.json` so it will persist across application restarts.
8.  You can enter new queries to continue interacting with the agent.

## Example Usage

1.  **User Query:** `What are the symptoms of the common cold?`

      - The agent might use `search_duck_duck_go` or `search_wikipedia` to find information about common cold symptoms.
      - It might then use `get_page_content` or `get_wikipedia_page` to get detailed information from a relevant search result or Wikipedia page.
      - The agent will then provide a response citing the source, summarizing the symptoms of the common cold based on the retrieved information.

2.  **User Query:** `Tell me about the history of the internet.`

      - The agent could use `search_wikipedia` to find a Wikipedia page about the "History of the Internet".
      - Then, use `get_wikipedia_page` to get the full content of the Wikipedia page.
      - Finally, the agent will provide a summary of the history of the internet, likely citing Wikipedia as the source.

## Potential Improvements and Future Work

  - **Enhanced Tool Selection Logic:** Improve the agent's ability to choose the most appropriate tool for a given query. This could involve more sophisticated natural language understanding and tool descriptions.
  - **More Diverse Tools:** Expand the toolset to include other search engines, specialized databases, APIs, or even local file access.
  - **Improved Citation Mechanism:** Implement a more robust citation system that clearly indicates which parts of the response come from which sources.
  - **Context Management:**  Enhance the agent's ability to maintain context across multiple turns of conversation for more complex and nuanced interactions.
  - **Error Handling and Fallback Strategies:**  Improve error handling and implement fallback strategies when tools fail or return inadequate responses.
  - **User Feedback and Refinement:** Incorporate user feedback mechanisms to continuously improve the agent's performance and tool utilization.
  - **Persist Conversation History in a Database:** Use a more robust database (like SQLite, PostgreSQL, etc.) for persistent conversation history instead of a simple JSON file.
