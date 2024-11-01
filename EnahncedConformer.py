import base64

import librosa
import numpy as np
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from starlette.requests import Request
import html
import re
import logging
import soundfile as sf
from nemo.utils import logging as nemo_logging
import nemo.collections.asr as nemo_asr
import requests
from io import BytesIO
import os  # To work with file paths
from fastapi.responses import JSONResponse

# Initialize FastAPI application
app = FastAPI()

# Set global logging level to suppress unnecessary logs
logging.getLogger().setLevel(logging.ERROR)
nemo_logging.setLevel(logging.ERROR)  # Suppress NeMo logs

# Load the ASR model (pre-trained)
model = nemo_asr.models.ASRModel.from_pretrained(model_name="Rangan00/Sanskrit_STT_enhanced")
model.eval()
model = model.to('cuda')  # Move the model to GPU if available

def compare_texts(predicted_text, ground_truth_text):
    """
    Compare predicted text with ground truth text using an external service.
    Returns highlighted differences in HTML format.
    """
    url = 'https://text-compare.com/'  # URL of the comparison service
    data = {'text1': predicted_text, 'text2': ground_truth_text, 'with_ajax': 1}
    response = requests.post(url, data=data)

    if response.status_code == 200:
        comparison_json = response.json()
        comparison_html = comparison_json.get('comparison', '')
        comparison_html = html.unescape(comparison_html)
        soup = BeautifulSoup(comparison_html, 'html.parser')
        difference_table = soup.find('table', {'class': 'text-compare'})

        if difference_table:
            highlighted_result = ""
            for row in difference_table.select('tr'):
                cells = row.find_all('td')
                if cells and len(cells) >= 3:
                    line_content = cells[2].prettify()
                    highlighted_result += line_content
            return highlighted_result
        else:
            return "No comparison results found."
    else:
        return f"Error: Unable to reach the service. Status code: {response.status_code}"

def write_html_output_to_file(content_with_td):
    """
    Extract <pre> content and wrap it for rendering in HTML.
    """
    pre_content = re.search(r'<pre>(.*?)</pre>', content_with_td, re.DOTALL).group(1)
    output_html = f"""<pre class="wrap-pre">{pre_content}</pre>"""
    return output_html

def transcribe_chunks(chunks, sr, model, batch_size=1, language_id='sa'):
    """
    Transcribe audio chunks using the ASR model.
    Returns combined transcription from all chunks.
    """
    combined_transcription = ""
    for i, chunk in enumerate(chunks):
        temp_wav = f'temp_chunk_{i}.wav'
        sf.write(temp_wav, chunk, sr)
        transcription = model.transcribe([temp_wav], batch_size=batch_size, logprobs=False, language_id=language_id)[0]
        combined_transcription += ' '.join(transcription) + " "
    return combined_transcription.strip()

def load_audio_from_bytes(audio_bytes, sample_rate=16000):
    """
    Load audio data from raw bytes and return audio array and sample rate.
    """
    audio, sr = librosa.load(BytesIO(audio_bytes), sr=sample_rate)
    return audio, sr

def chunk_audio(audio, sr, chunk_duration=120):
    """
    Chunk the audio into segments of specified duration.
    Returns a list of audio chunks.
    """
    chunk_length = chunk_duration * sr
    num_chunks = int(np.ceil(len(audio) / chunk_length))
    chunks = [audio[i * chunk_length:(i + 1) * chunk_length] for i in range(num_chunks)]
    return chunks


def postprocess_html(html_content):
    """
    Remove highlights on standalone whitespaces, remove any highlighted trailing or leading whitespace,
    and handle cases where whitespace may be highlighted at the start or end of words.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Pattern to match words that contain or end with 'ं' or 'म्' (like "कृष्णं", "कृष्णम्") or only whitespace
    unwanted_highlight_pattern = re.compile(r'^\s*$|.*[ं्]$')

    # Iterate over all spans with the 'difference' class
    for span in soup.find_all('span', {'class': 'difference'}):
        span_text = span.get_text()

        # Strip leading/trailing whitespace for comparison
        if span_text.isspace() or unwanted_highlight_pattern.match(span_text.strip()):
            # Remove the highlight if it is entirely whitespace or matches unwanted patterns
            span.unwrap()
        else:
            # Strip only the highlighted whitespace within spans if needed
            stripped_text = span_text.strip()
            if stripped_text != span_text:
                # Replace content with stripped text to remove unwanted highlighted whitespace
                span.string = stripped_text

    # Merge consecutive text nodes to handle any residual whitespaces
    for element in soup.find_all(text=True):
        if element.next_sibling and isinstance(element.next_sibling, str):
            element.replace_with(element + element.next_sibling)

    return str(soup)



# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the home page with the input form.
    """
    return templates.TemplateResponse("index.html", {"request": request})


import base64  # Add this import at the top of your file

import base64  # Import base64


@app.post("/process/")
async def process_audio_and_transcription(
        request: Request,
        audio_file: UploadFile = File(None),  # Accept audio file input, optional
        recorded_audio: str = Form(None),  # Accept recorded audio input, optional
        canto: int = Form(...),
        chapter: int = Form(...)):
    # Check if audio_file or recorded_audio is provided
    audio, sr = None, None  # Initialize audio and sample rate variables

    if recorded_audio:
        # Decode base64 audio
        header, encoded = recorded_audio.split(",", 1)  # Split the header
        audio_bytes = base64.b64decode(encoded)

        # Read the Opus audio data using pydub
        try:
            audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
            audio = np.array(audio_segment.get_array_of_samples())
            sr = audio_segment.frame_rate
            print("Successfully loaded recorded audio.")
        except Exception as e:
            return {"error": f"Error loading audio: {str(e)}"}

    elif audio_file:
        # Detect and handle .m4a format
        if audio_file.filename.endswith(".m4a"):
            audio_bytes = await audio_file.read()
            try:
                # Convert .m4a to wav using pydub
                audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="m4a")
                audio = np.array(audio_segment.get_array_of_samples())
                sr = audio_segment.frame_rate
                print("Successfully loaded .m4a file as wav.")
            except Exception as e:
                return {"error": f"Error converting .m4a audio: {str(e)}"}
        else:
            audio_bytes = await audio_file.read()
            try:
                audio, sr = load_audio_from_bytes(audio_bytes, sample_rate=16000)
                print("Successfully loaded uploaded audio.")
            except Exception as e:
                return {"error": f"Error loading audio: {str(e)}"}

    else:
        return {"error": "No audio file provided."}

    # Construct transcription file path from canto and chapter
    transcription_file = f"dataset/{canto}_transcriptions/{chapter}.txt"

    # Check if the transcription file exists
    if not os.path.exists(transcription_file):
        return {"error": f"Transcription file {transcription_file} not found."}

    # Load transcription text
    with open(transcription_file, 'r', encoding='utf-8') as file:
        ground_truth_text = file.read().replace("\n", "")

    # Split the audio into chunks and transcribe
    chunks = chunk_audio(audio, sr, chunk_duration=120)
    combined_transcription = transcribe_chunks(chunks, sr, model, batch_size=1, language_id='sa')

    # Perform comparison and get highlighted differences
    highlighted_html = compare_texts(combined_transcription, ground_truth_text)

    # Postprocess the highlighted HTML to remove unwanted highlights
    processed_html = postprocess_html(highlighted_html)

    result = write_html_output_to_file(processed_html)

    # Check if the request was made with AJAX
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        # Return JSON response with the result HTML for AJAX
        return JSONResponse(content={"result_html": result})

    # Render full template if not AJAX
    return templates.TemplateResponse("index.html", {"request": request, "result": result})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
