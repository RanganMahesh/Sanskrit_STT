from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from fuzzy import Soundex, nysiis
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from Levenshtein import distance as levenshtein_distance

app = FastAPI()

# Load templates
templates = Jinja2Templates(directory="templates")

# Load pre-trained processor and model
# Load model directly
from transformers import AutoProcessor, AutoModelForCTC

processor = AutoProcessor.from_pretrained("Rangan00/Sanskrit_STT")
model = AutoModelForCTC.from_pretrained("Rangan00/Sanskrit_STT").to('cuda')


def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    if sr != 16000:
        y = librosa.resample(y, sr, 16000)
    audio_tensor = torch.tensor(y).unsqueeze(0).to('cuda')
    return audio_tensor


def fuzzy_match_transcription(prediction, ground_truth):
    predicted_transliteration = transliterate(prediction, sanscript.DEVANAGARI, sanscript.ITRANS)
    ground_truth_transliteration = transliterate(ground_truth, sanscript.DEVANAGARI, sanscript.ITRANS)

    soundex = Soundex(4)
    predicted_soundex = soundex(predicted_transliteration)
    ground_truth_soundex = soundex(ground_truth_transliteration)

    predicted_metaphone = nysiis(predicted_transliteration)
    ground_truth_metaphone = nysiis(ground_truth_transliteration)

    # Calculate Levenshtein distance between phonetic representations
    distance = levenshtein_distance(predicted_metaphone, ground_truth_metaphone)
    max_len = max(len(predicted_metaphone), len(ground_truth_metaphone))
    similarity_score = 1 - (distance / max_len)

    return {
        "predicted_transliteration": predicted_transliteration,
        "ground_truth_transliteration": ground_truth_transliteration,
        "similarity_score": similarity_score
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/test")
async def test():
    return {"message": "Test endpoint is working"}

@app.post("/process/")
async def process_audio_and_transcription(
        request: Request,
        audio_path: str = Form(...),
        transcription_path: str = Form(...)):
    # Load and preprocess audio
    audio_tensor = preprocess_audio(audio_path)

    # Load transcription text file and remove newlines
    with open(transcription_path, 'r', encoding='utf-8') as file:
        ground_truth_text = file.read().replace("\n", "")

    # Transcribe audio using the model
    with torch.no_grad():
        logits = model(input_values=audio_tensor).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    # Perform fuzzy matching
    result = fuzzy_match_transcription(transcription, ground_truth_text)

    # Render result on the same page
    return templates.TemplateResponse("index.html", {
        "request": request,
        "transcription": transcription,
        "ground_truth_transcription": ground_truth_text,
        "similarity_result": result
    })
