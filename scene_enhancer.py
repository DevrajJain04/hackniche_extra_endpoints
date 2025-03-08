from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import torch
import nltk
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import sent_tokenize
from collections import Counter

# Download NLTK resources if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv()

# Load Hugging Face model for efficiency
MODEL_NAME = "distilgpt2"
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
hf_model.eval()

# Configure API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY environment variable is not set")
else:
    genai.configure(api_key=api_key)

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
scene_reports = []

app = FastAPI(
    title="Script Analysis and Generation API",
    description="An API for analyzing and generating script content",
    version="2.0.0"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class ScriptInput(BaseModel):
    script_text: str

class TextInput(BaseModel):
    script_text: str

class StyleAnalysisRequest(BaseModel):
    excerpts: list[str]

class TextAnalysisRequest(BaseModel):
    text_samples: list[str]

class SceneGenerationRequest(BaseModel):
    narrative_direction: str
    previous_scenes: list[str] = []  # Optional with default empty list

def analyze_script(script_content):
    """Analyzes the script and ensures JSON formatted output."""
    prompt = """
    Analyze the provided movie scene and return the results in the following strict JSON format. Your goal is to summarize key elements that capture the essence of the scene, while keeping the description, plot points, and character emotions concise. Pay attention to the mood, characters' emotions, and key visual and auditory cues. Avoid extra commentary and ensure the JSON is formatted correctly. Here's the structure you should follow:

    ```json
    {
        "scene_description": "",
        "sound_effects": [],
        "visual_cues": [],
        "characters": [
            {
                "name": "",
                "emotion": "",
                "description": ""
            }
        ],
        "readability_score": "",
        "summary": "",
        "poetic_devices": [
            {
                "device": "",
                "example": ""
            }
        ],
        "narrative_direction": ""
    }

    Script:
    {script_content}
    
    JSON Output:
    """
    prompt = prompt.replace("{script_content}", script_content)
    
    try:
        response = gemini_model.generate_content(prompt)
        
        # Ensure clean JSON output
        try:
            # Handle both direct JSON responses and responses with markdown code blocks
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.strip("```json").strip("```").strip()
            json_output = json.loads(text)
            return json_output
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid JSON response from AI model: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

@app.get("/")
def root():
    return {"message": "Writing Style Analyzer API is running."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
def analyze_script_endpoint(script: ScriptInput):
    print("Analyze endpoint called")
    if not script.script_text.strip():
        raise HTTPException(status_code=400, detail="Script text cannot be empty.")
    
    analysis = analyze_script(script.script_text)
    scene_reports.append(analysis)  # Append new scene analysis
    
    return {"analysis": analysis, "scene_count": len(scene_reports)}

@app.get("/report")
def get_aggregated_report():
    print("Report endpoint called")
    if not scene_reports:
        return {"report": "No scenes analyzed yet."}

    # Collect all characters and their attributes from all scenes
    aggregated_characters = {}
    narrative_directions = []
    
    for scene in scene_reports:
        # Aggregate characters and their attributes
        for char in scene.get("characters", []):
            name = char["name"]
            if name not in aggregated_characters:
                aggregated_characters[name] = {"emotions": set(), "descriptions": set()}
            
            aggregated_characters[name]["emotions"].add(char["emotion"])
            aggregated_characters[name]["descriptions"].add(char["description"])
        
        # Collect narrative direction history
        narrative_directions.append(scene.get("narrative_direction", ""))

    # Convert sets to lists for JSON compatibility
    for name in aggregated_characters:
        aggregated_characters[name]["emotions"] = list(aggregated_characters[name]["emotions"])
        aggregated_characters[name]["descriptions"] = list(aggregated_characters[name]["descriptions"])

    return {
        "total_scenes": len(scene_reports),
        "characters": aggregated_characters,
        "narrative_directions": narrative_directions
    }

@app.delete("/reset")
def reset_report():
    global scene_reports
    scene_reports = []
    return {"message": "Scene report reset successful."}

@app.get("/stats")
def get_script_stats():
    print("Stats endpoint called")
    if not scene_reports:
        return {"stats": {"character_count": 0, "character_names": [], "emotions": {}}}
    
    characters = scene_reports[-1].get("characters", [])
    character_names = [char["name"] for char in characters]
    emotions = {char["name"]: char["emotion"] for char in characters}

    stats = {
        "character_count": len(characters),
        "character_names": character_names,
        "emotions": emotions
    }

    return {"stats": stats}

@app.get("/readability")
def get_readability_score():
    print("Readability endpoint called")
    if not scene_reports:
        return {"readability_score": 0}
    return {"readability_score": scene_reports[-1].get("readability_score", 0)}

@app.get("/narrative_direction")
def get_narrative_direction():
    print("Narrative direction endpoint called")
    if not scene_reports:
        return {"narrative_direction": "No analysis available"}
    return {"narrative_direction": scene_reports[-1].get("narrative_direction", "No analysis available")}

@app.get("/poetic_devices")
def get_poetic_devices():
    print("Poetic devices endpoint called")
    if not scene_reports:
        return {"poetic_devices": []}
    return {"poetic_devices": scene_reports[-1].get("poetic_devices", [])}


@app.post("/analyze-style")
def analyze_writing_style(request: StyleAnalysisRequest):
    """Analyzes writing style from provided text samples."""
    print("Analyze style endpoint called")
    if not request.excerpts:
        raise HTTPException(status_code=400, detail="No excerpts provided.")
        
    text = " ".join(request.excerpts)
    sentences = sent_tokenize(text)
    words = re.findall(r'\b\w+\b', text.lower())
    
    if not sentences or not words:
        raise HTTPException(status_code=400, detail="Invalid text input.")
    
    avg_sentence_length = sum(len(re.findall(r'\b\w+\b', s)) for s in sentences) / len(sentences)
    vocabulary_diversity = len(set(words)) / len(words)
    punctuation_freq = dict(Counter(re.findall(r'[.,!?;:"\'-]', text)))  # Convert to dict for JSON
    
    response = {
        "avg_sentence_length": avg_sentence_length,
        "vocabulary_diversity": vocabulary_diversity,
        "punctuation_frequency": punctuation_freq,
        "sentence_count": len(sentences)
    }
    global writing_style 
    writing_style = response

    return response

@app.get("/generate-scene")  # Fixed the endpoint name to use hyphen
def generate_scene():
    """Generates a new scene based on narrative direction."""
    print("Generate scene endpoint called")
    try:
        previous_context = scene_reports[-1].get("summary", "") if scene_reports else ""
        
        prompt = f"""
        Previous scenes: {previous_context}
        writing_style of author: {writing_style}
        Narrative direction: {[scene.get("narrative_direction","") for scene in scene_reports]}
        
        Please generate a creative and compelling movie scene based on the narrative direction above.
        The scene should have realistic dialogue, vivid descriptions, and strong emotional content.
        Include both character interactions and environmental details.
        if none provided return empty string
        """
        
        # Generate the scene using Gemini
        response = gemini_model.generate_content(prompt)
        generated_scene = response.text
        
        return {"generated_scene": generated_scene}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scene generation failed: {str(e)}")

# Note: Your original filename needs to be 'scene_enhancer.py' for this to work
# uvicorn scene_enhancer:app --host 0.0.0.0 --port 8000