
# 🌿 YouTube Summarizer with Llama 3 🌿

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Available Languages](#available-languages)
- [Customization](#customization)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
The YouTube Summarizer with Llama 3 is a tool that allows users to easily extract information, generate transcriptions, and produce detailed summaries from YouTube videos. By leveraging the power of `LangChain`, `Ollama`, and `pytube`, it supports a wide range of languages to deliver high-quality and multilingual summaries.

## Features
- Extracts video details (title and description) from YouTube URLs.
- Transcribes YouTube videos using LangChain's document loaders.
- Summarizes the transcription using the Llama 3 model.
- Supports multi-language summaries with automatic translation.
- Provides a user-friendly UI with custom CSS for a clean, modern look.
- Adjustable chunk size, overlap size, and temperature settings for better control over text processing.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- `pip` package manager
- `Ollama` installed locally and running on `http://localhost:11434`
- Access to OpenAI models (e.g., `gpt-4`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/motolomygolda/MultiLingual-YouTube-Summarizer-using-LLAMA3.git
    cd MultiLingual-YouTube-Summarizer-using-LLAMA3
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure Ollama is running on your local machine:
    - Download and install [Ollama](https://ollama.com).
    - Run the server using:
      ```bash
      ollama start
      ```

## Usage
1. Run the application:
    ```bash
    python main.py
    ```
   
2. Access the UI:
   - Open your browser and go to `http://localhost:7860`.
   
3. How to use the interface:
   - Enter a YouTube URL in the text box.
   - Click "Get Info" to fetch the video title and description.
   - Click "Get Transcription" to extract the transcript and token count.
   - Adjust settings (e.g., temperature, chunk size, overlap size, language).
   - Click "Summarize" to generate a summary.

## Available Languages
The summarizer supports the following languages:
- African Languages: Amharic, Hausa, Kinyarwanda, Somali, Swahili, Tigrinya, Twi, Wolof, Yoruba, Zulu, Oromo.
- Indian Languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.
- European Languages: English, French, German, Greek, Italian, Portuguese, Spanish, Swedish, Dutch, Danish, Finnish, Norwegian, Polish, Romanian, Russian, Ukrainian, Welsh.
- Asian Languages: Arabic, Chinese (Simplified & Traditional), Hebrew, Japanese, Korean, Persian, Thai, Vietnamese.
- Others: Filipino, Indonesian, Catalan, Slovak, Slovenian, Croatian, Lithuanian, Latvian, Hungarian, Icelandic, Estonian, Czech, Bulgarian.

## Customization
If you want to change the application's look, modify the `custom_css` section in the script to update the background colors, button styles, and other UI elements.

## How It Works
### 1. Extract Video Details
- Uses `pytube` to fetch the video title.
- Retrieves the video description using a regex-based approach.

### 2. Transcription
- Leverages `LangChain` with the `YoutubeLoader` to extract the full video transcript.
- Uses `RecursiveCharacterTextSplitter` to split long transcripts into manageable chunks.

### 3. Summarization
- The `Ollama` Llama 3 model processes the text chunks using a map-reduce approach.
- Custom prompt templates help generate detailed summaries.
- Optionally translates the summary using `GoogleTranslator`.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please create an issue or submit a pull request.

### To Do:
- [ ] Improve UI with additional animations and design elements.
- [ ] Add support for more language models.
- [ ] Optimize the summarization process for longer videos.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
