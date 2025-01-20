# llm-pdf-summarization
Using LLM to summarize PDF. 

Try out the [Web demo](https://huggingface.co/spaces/hermanda/pdf-summarizer), integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). 

## Usage

### Local
```
uv venv --python=3.12
source .venv/bin/activate
python app.py
```

### Docker
```
docker build -t pdf-summarizer-app .
docker run -p 3000:3000 --name pdf-summarizer -e OPENAI_API_KEY=your_openai_api_key_here pdf-summarizer-app
```
Now you can enjoy the app on http://localhost:3000. 

To remove the image
```
docker rm pdf-summarizer
```
