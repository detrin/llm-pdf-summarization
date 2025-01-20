import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import os
from dotenv import load_dotenv

os.makedirs("data", exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def summarize_pdf(pdf_file, custom_prompt="", openai_api_key=None):
    """
    Summarizes the content of a PDF file using a custom prompt.

    Args:
        pdf_file (UploadedFile): The uploaded PDF file.
        custom_prompt (str): The prompt for summarization.
        openai_api_key (str, optional): User-provided OpenAI API key.

    Returns:
        tuple: Summary in markdown format and the cost in USD.
    """
    pdf_path = os.path.join("data", "tmp.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_file)

    api_key = openai_api_key if openai_api_key else OPENAI_API_KEY
    
    if not api_key:
        return "Error: No OpenAI API key provided.", "N/A"

    with get_openai_callback() as cb:
        try:
            model = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0,
                openai_api_key=api_key
            )

            loader = PyPDFLoader(pdf_path)
            docs = loader.load_and_split()
            
            if not custom_prompt.strip():
                custom_prompt = default_prompt

            prompt_template = (
                custom_prompt
                + """

            {text}

            SUMMARY:"""
            )
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(
                model, 
                chain_type="map_reduce", 
                map_prompt=PROMPT, 
                combine_prompt=PROMPT
            )
            summary = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
            total_cost = cb.total_cost

            return summary, f"${total_cost:.4f}"
        
        except Exception as e:
            return f"An error occurred: {str(e)}", "N/A"

default_prompt = (
    "Summarize this paper. Return markdown, keep it in a language that scientists understand, "
    "but the purpose is to highlight the key takeaways, so that we save time for the reader."
)

with gr.Blocks() as demo:
    gr.Markdown("# PDF Summarizer 📝")
    gr.Markdown("Upload a PDF, customize your summarization prompt, and get a concise summary along with the processing cost.")

    with gr.Row():
        with gr.Column():
            if OPENAI_API_KEY is None:
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API key."
                )
            else:
                api_key_input = gr.Textbox(
                    label="OpenAI API Key (Optional)",
                    type="password",
                    placeholder="Enter your OpenAI API key if you want to override the global key."
                )
            prompt_input = gr.Textbox(
                label="Custom Prompt",
                lines=4,
                value=default_prompt,
                placeholder="Enter your custom summarization prompt here..."
            )
            pdf_input = gr.File(
                label="Upload PDF",
                type="binary",
                file_types=[".pdf"],
            )
            summarize_btn = gr.Button("Summarize")
        
        with gr.Column():
            cost_output = gr.Textbox(label="Approximate Cost (USD)", interactive=False)
            summary_output = gr.Markdown(label="Summary")
            
    
    summarize_btn.click(
        fn=summarize_pdf,
        inputs=[pdf_input, prompt_input, api_key_input],
        outputs=[summary_output, cost_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("Created by [Daniel Herman](https://www.hermandaniel.com)")

# Launch the app
if __name__ == "__main__":
    demo.launch()