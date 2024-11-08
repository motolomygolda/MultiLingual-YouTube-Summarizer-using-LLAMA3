import pytube
import requests
import re
import gradio as gr
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import tiktoken
from deep_translator import GoogleTranslator
from langchain.prompts import PromptTemplate


# Function to get YouTube description using regex parsing.
def get_youtube_description(url: str):
    full_html = requests.get(url).text
    y = re.search(r'shortDescription":"', full_html)
    desc = ""
    count = y.start() + 19  # adding the length of the 'shortDescription":"'
    while True:
        letter = full_html[count]
        if letter == "\"":
            if full_html[count - 1] == "\\":
                desc += letter
                count += 1
            else:
                break
        else:
            desc += letter
            count += 1
    return desc

# Function to get YouTube video info (title and description).
def get_youtube_info(url: str):
    yt = pytube.YouTube(url)
    title = yt.title if yt.title else "None"
    desc = get_youtube_description(url) or "None"
    return title, desc

# Function to get YouTube transcript using LangChain.
def get_youtube_transcript_loader_langchain(url: str):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    return loader.load()

# Wrap documents into a single string.
def wrap_docs_to_string(docs):
    return " ".join([doc.page_content for doc in docs]).strip()

# Function to split text into chunks.
def get_text_splitter(chunk_size: int, overlap_size: int):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=overlap_size
    )

# Get the full transcript from a YouTube video.
def get_youtube_transcription(url: str):
    text = wrap_docs_to_string(get_youtube_transcript_loader_langchain(url))
    enc = tiktoken.encoding_for_model("gpt-4")
    count = len(enc.encode(text))
    
    return text, count

# Define prompts using PromptTemplate for map_reduce chain.
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""You are a summarization assistant. Summarize the following text in as much detail as possible. Include all the main ideas, key points, and important details. Provide a thorough and complete summary of the content. Text: {text}"""
)

combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="""You now have multiple summaries. Combine all of these summaries into one comprehensive, detailed, and thorough final summary. Include all the main ideas and key points from each chunk. The final summary should provide a detailed and full understanding of the original content. Summaries: {text}"""
)

# Function to generate a summary with map_reduce.
def get_transcription_summary(url: str, temperature: float, chunk_size: int, overlap_size: int, language: str):
    docs = get_youtube_transcript_loader_langchain(url)
    
    # Adjust chunk size for larger text processing.
    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    split_docs = text_splitter.split_documents(docs)

    # Initialize the model with adjusted temperature.
    llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=temperature)

    # Load map_reduce chain with customized prompts.
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt
    )

    # Generate the summary.
    output = chain.invoke(split_docs)
    summary = output['output_text']
    
    
    # Translate summary if language is not English.
    if language != 'English':
        translated_summary = GoogleTranslator(source='auto', target=language.lower()).translate(summary)
        return translated_summary

    return summary



# Custom CSS styling to enhance the UI with green colors
custom_css = """
    body {
        background-color: #f0f9f4; /* Light green background */
    }
    .gr-textbox, .gr-number, .gr-dropdown {
        border-color: #48bb78; /* Green border for inputs */
    }
    .gr-button-primary {
        background-color: #38a169; /* Green for primary buttons */
        color: #ffffff;
    }
    .gr-button-primary:hover {
        background-color: #2f855a; /* Darker green on hover */
    }
    .gr-button-stop {
        background-color: #e53e3e; /* Red stop button */
        color: #ffffff;
    }
    .gr-panel {
        background: #ffffff; /* White panel background */
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""# 🌿 **YouTube Summarizer with Llama 3** 🌿  
    Enter a YouTube URL, get details, and generate transcription and summary easily!  
    """)
    with gr.Row(equal_height=True) as r0:
        with gr.Column(scale=4) as r0c1:
            url = gr.Textbox(label='Enter the YouTube URL', value="", placeholder="https://youtube.com/watch?v=...")
        with gr.Column(scale=1) as r0c2:
            bttn_info_get = gr.Button('Get Info', variant='primary')
            bttn_clear = gr.ClearButton(interactive=True, variant='stop')

    with gr.Row(variant='panel') as r1:
        with gr.Column(scale=2) as r1c1:
            title = gr.Textbox(label='Title', lines=2, max_lines=10, show_copy_button=True)
        with gr.Column(scale=3) as r1c2:
            desc = gr.Textbox(label='Description', lines=2, max_lines=10, autoscroll=False, show_copy_button=True)
            bttn_info_get.click(fn=get_youtube_info, inputs=url, outputs=[title, desc], api_name="get_youtube_info")

    with gr.Row(equal_height=True) as r2:        
        with gr.Column() as r2c1:
            bttn_trns_get = gr.Button("Get Transcription", variant='primary')
            tkncount = gr.Number(label='Token Count (est)', interactive=False)
        with gr.Column() as r2c3:
            bttn_summ_get = gr.Button("Summarize", variant='primary')
            with gr.Row():
                with gr.Column(scale=1, min_width=95):
                    temperature = gr.Number(label='Temperature', minimum=0.0, step=0.01, precision=-2, value=0.3)
                with gr.Column(scale=1, min_width=95):
                    chunk = gr.Number(label='Chunk Size', minimum=200, step=100, value=5000)
                with gr.Column(scale=1, min_width=95):
                    overlap = gr.Number(label='Overlap Size', minimum=0, step=10, value=100)
                with gr.Column(scale=1, min_width=125):
                    language = gr.Dropdown(label="Language",  choices=[
                            "Amharic", "Arabic", "Akan", "Bengali", "Bhojpuri", "Bulgarian", "Catalan", 
                            "Chinese (Simplified)", "Croatian", "Czech", "Danish", 
                            "Dutch", "English", "Estonian", "Filipino", "Finnish", "French", "German", 
                            "Greek", "Gujarati", "Hausa", "Hebrew", "Hindi", "Hungarian", "Icelandic", 
                            "Igbo", "Indonesian", "Italian", "Japanese", "Kannada", "Kinyarwanda", 
                            "Korean", "Latvian", "Lithuanian", "Luo", "Malay", "Malayalam", "Marathi", 
                            "Nepali", "Norwegian", "Odia", "Oromo", "Persian", "Polish", "Portuguese", 
                            "Punjabi", "Romanian", "Russian", "Slovak", "Slovenian", "Somali", "Spanish", 
                            "Swahili", "Swedish", "Tamil", "Telugu", "Thai", "Tigrinya", "Turkish", 
                            "Twi", "Ukrainian", "Urdu", "Vietnamese", "Welsh", "Wolof", "Xhosa", "Yoruba", 
                            "Zulu"
                        ], value="English")

    with gr.Row() as r3:
        with gr.Column() as r3c1:
            trns_raw = gr.Textbox(label='Transcript', show_copy_button=True)
        with gr.Column() as r3c2:
            trns_sum = gr.Textbox(label="Summary", show_copy_button=True)
    
    bttn_trns_get.click(fn=get_youtube_transcription, inputs=url, outputs=[trns_raw, tkncount])
    bttn_summ_get.click(fn=get_transcription_summary, inputs=[url, temperature, chunk, overlap, language], outputs=trns_sum)
    
    bttn_clear.add([url, title, desc, trns_raw, trns_sum, tkncount])


if __name__ == "__main__":
    demo.launch(share=True, server_name="localhost", server_port=7860) 

