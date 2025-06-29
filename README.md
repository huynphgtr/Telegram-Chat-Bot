# Telegram AI Chatbot

This project is a versatile Telegram bot designed to provide various conversational experiences. It includes basic predefined responses, integrates with OpenAI's GPT models for Retrieval-Augmented Generation (RAG) capabilities, and can also leverage a local Llama model for text generation.

-----

## Features

  * **Basic Commands:** Responds to standard Telegram commands like `/start`, `/help`, and `/custom`.
  * **Predefined Responses:** Handles simple conversational queries with hardcoded replies (e.g., "hello," "how are you").
  * **OpenAI GPT Integration:**
      * **Direct Chat:** Can interact with OpenAI's GPT-3.5-turbo for general conversational AI.
      * **Retrieval-Augmented Generation (RAG):** Answers questions based on content extracted from PDF documents. This involves:
          * Reading text from PDF files.
          * Chunking the text into smaller segments.
          * Generating embeddings for each chunk using OpenAI's `text-embedding-ada-002` model.
          * Finding relevant chunks based on user queries using cosine similarity.
          * Constructing a context-aware prompt for the GPT model to generate informed responses.
          * Caching embeddings to speed up subsequent processing.
  * **Local Llama Model Integration:** Generates responses using a locally loaded `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model via the `transformers` library. This allows for offline AI chat capabilities.

-----

## Prerequisites

Before running the bot, ensure you have the following installed:

  * **Python 3.8+**
  * **Required Python Libraries:** You'll need to install the libraries listed in the `requirements.txt` (see below for content).
  * **Telegram Bot Token:** Obtain a bot token from BotFather on Telegram.
  * **OpenAI API Key:** If you plan to use the GPT or RAG features.
  * **PDF File:** For the RAG functionality, you'll need a PDF file (e.g., named `your_document.pdf`) that the bot will process for generating context-aware responses.

-----

## Setup and Installation

1.  **Clone the Repository (or save the code):**

    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

    (If you don't have a repository, just save the provided Python code as `bot.py` in a new directory.)

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` in your project directory and add the following:

    ```
    python-telegram-bot
    openai
    PyMuPDF # for fitz
    scikit-learn # for cosine_similarity
    numpy
    torch
    transformers
    ```

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure API Keys and Bot Username:**
    Open your `bot.py` file and fill in your actual `TOKEN`, `BOT_USERNAME`, and `openai.api_key`.

    ```python
    TOKEN: Final = 'YOUR_TELEGRAM_BOT_TOKEN'
    BOT_USERNAME: Final = '@YourBotUsername' # e.g., @MyAwesomeAIBot
    # ...
    openai.api_key = "YOUR_OPENAI_API_KEY"
    # ...
    client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
    ```

    **Important:** Remember to replace `'YOUR_TELEGRAM_BOT_TOKEN'`, `'@YourBotUsername'`, and `"YOUR_OPENAI_API_KEY"` with your actual credentials. For the `BOT_USERNAME`, make sure to include the `@` symbol.

6.  **Place Your PDF File:**
    Put the PDF file you want to use for RAG (e.g., `my_document.pdf`) in the same directory as your `bot.py` file. Then, update the `process_pdf` function call with the correct file path:

    ```python
    # In bot.py, update this line with your PDF file name
    chunks, chunk_embeddings = process_pdf("my_document.pdf") # Replace "my_document.pdf"
    ```

-----

## Running the Bot

To start the bot, simply run the Python script:

```bash
python bot.py
```

You'll see "Start polling..." and "Polling..." messages in your console, indicating the bot is active.

-----

## Usage

Once the bot is running, you can interact with it on Telegram:

  * **Commands:**
      * `/start`: Greets you.
      * `/help`: Provides a help message.
      * `/custom`: A custom command response.
  * **Basic Chat:**
      * Type "hello", "how are you", "who are you", etc., for predefined responses.
  * **AI Chat (General):**
      * The `handle_ai_message` function is intended for RAG. If you want a general OpenAI chat, you'd integrate the `chat_with_gpt` function directly into a message handler.
  * **AI Chat (RAG - from PDF):**
      * The bot will attempt to answer questions based on the content of the PDF you provided. Ask questions related to the document's content (e.g., "What is mentioned about [topic] in the document?").
      * *Note:* There are two `handle_ai_message` functions in your code. Ensure only one is registered with a `MessageHandler` and that it uses `get_rag_response` if you want to enable RAG for all text messages.
  * **AI Chat (Llama Model):**
      * To use the `TinyLlama` model, you would need to register the `handle_model_message` as a `MessageHandler`. Currently, `handle_message` is the primary handler.

**Important Note on Message Handlers:**
In your current `if __name__ == '__main__':` block, `app.add_handler(MessageHandler(filters.TEXT, handle_message))` means all text messages go through `handle_message`. To use the RAG or Llama model, you'll need to modify the `MessageHandler` registration.

**Example modification for RAG:**
To make all general text messages go through the RAG pipeline:

```python
if __name__ == '__main__':
    print('Start polling...')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))

    # Messages - All text messages now go through RAG
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ai_message)) # Updated handler

    # Error handling
    app.add_error_handler(error)

    print('Polling...')
    app.run_polling(poll_interval=3)
```

*(Similarly, if you want to use `handle_model_message` instead, replace `handle_ai_message` with `handle_model_message`.)*

-----

## Contributing

Feel free to fork this project, open issues, or submit pull requests.

-----
