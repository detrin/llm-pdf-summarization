import os
from typing import Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI


os.makedirs("data", exist_ok=True)
load_dotenv()
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")


def summarize_pdf(
    pdf_file: bytes, custom_prompt: str = "", openai_api_key: Optional[str] = None
) -> Tuple[str, str]:
    """
    Summarizes the content of a PDF file using a custom prompt.

    Args:
        pdf_file (bytes): The uploaded PDF file as bytes.
        custom_prompt (str): The prompt for summarization.
        openai_api_key (Optional[str]): User-provided OpenAI API key.

    Returns:
        Tuple[str, str]: Summary in markdown format and the cost in USD.
    """
    pdf_path: str = os.path.join("data", "tmp.pdf")
    try:
        with open(pdf_path, "wb") as f:
            f.write(pdf_file)
    except IOError as e:
        return f"Failed to write PDF file: {e}", "N/A"

    api_key: Optional[str] = openai_api_key or OPENAI_API_KEY

    if not api_key:
        return "Error: No OpenAI API key provided.", "N/A"

    with get_openai_callback() as callback:
        try:
            model = ChatOpenAI(
                model="gpt-4o-mini",  # Verify the correct model name
                temperature=0.0,
                openai_api_key=api_key,
            )

            loader = PyPDFLoader(pdf_path)
            documents = loader.load_and_split()

            prompt_text: str = custom_prompt.strip() or default_prompt
            prompt_template: str = f"{prompt_text}\n\n{{text}}\n\nSUMMARY:"
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

            summarize_chain = load_summarize_chain(
                llm=model,
                chain_type="map_reduce",
                map_prompt=prompt,
                combine_prompt=prompt,
            )

            chain_input = {"input_documents": documents}
            result = summarize_chain(chain_input, return_only_outputs=True)
            summary: str = result.get("output_text", "No summary generated.")
            total_cost: float = callback.total_cost

            return summary, f"${total_cost:.4f}"

        except Exception as e:
            return f"An error occurred during summarization: {str(e)}", "N/A"


default_prompt: str = (
    "Summarize this paper. Return markdown, keep it in a language that scientists understand, "
    "but the purpose is to highlight the key takeaways, so that we save time for the reader."
)
with gr.Blocks() as demo:
    gr.Markdown("# PDF Summarizer 📝")
    gr.Markdown(
        "Upload a PDF, customize your summarization prompt, and get a concise summary along with the processing cost."
    )

    with gr.Row():
        with gr.Column():
            api_key_label: str
            placeholder_text: str

            if OPENAI_API_KEY is None:
                api_key_label = "OpenAI API Key"
                placeholder_text = "Enter your OpenAI API key."
            else:
                api_key_label = "OpenAI API Key (Optional)"
                placeholder_text = (
                    "Enter your OpenAI API key if you want to override the global key."
                )

            api_key_input = gr.Textbox(
                label=api_key_label,
                type="password",
                placeholder=placeholder_text,
            )
            prompt_input = gr.Textbox(
                label="Custom Prompt",
                lines=4,
                value=default_prompt,
                placeholder="Enter your custom summarization prompt here...",
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
        outputs=[summary_output, cost_output],
    )

    gr.Markdown("---")
    gr.Markdown("Created by [Daniel Herman](https://www.hermandaniel.com), check out the code [detrin/llm-pdf-summarization](https://github.com/detrin/llm-pdf-summarization).")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=3000)
