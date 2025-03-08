from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
load_dotenv()

# Configure API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # Or your preferred way to store your key.

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash-lite')

app = FastAPI(
    title="FastAPI Boilerplate",
    description="A simple FastAPI boilerplate with CORS support and health check.",
    version="2.0.0"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScriptInput(BaseModel):
    script_text: str
class TextInput(BaseModel):
    script_text: str

cached_analysis = None  # Store the last analyzed script response

def analyze_script(script_content):
    """Analyzes the script and ensures JSON formatted output."""
    prompt = f"""
    Analyze the following scene and strictly return a well-formed JSON object with this structure:
    // keep the threshold for something to be considered an emotion or a description to be low
    ```json
    {{
        "scene_description": "Detailed description of the scene",
        "sound_effects": ["List of relevant sound effects"],
        "visual_cues": ["List of environmental and visual elements"],
        "characters": [
            {{
                "name": "Character name",
                "emotion": "Character emotions",
                "description": (upto 2 sentences at a maximum)
            }}
        ]
        "readability_score": "Flesch-Kincaid readability score",
        "sentiment": "Sentiment of the scene",
        "poetic_devices": {
            "example_device": {
                "count": "Number of times used",
                "examples": ["List of examples"]
            }
        },
        "narrative_direction":[very concise summary of ]
    }}
    ```
    Ensure the response is always valid JSON, with no additional text or commentary.
    
    Script:
    {script_content}
    
    JSON Output:
    """
    
    response = model.generate_content(prompt)
    
    # Ensure clean JSON output
    try:
        json_output = json.loads(response.text.strip().strip("```json").strip("```"))
        return json_output
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from AI model.")

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
def analyze_script_endpoint(script: ScriptInput):
    global cached_analysis
    if not script.script_text.strip():
        raise HTTPException(status_code=400, detail="Script text cannot be empty.")
    
    cached_analysis = analyze_script(script.script_text)
    return {"analysis": cached_analysis}

@app.get("/stats")
def get_script_stats():
    if cached_analysis is None:
        return {"stats": {"character_count": 0, "character_names": [], "emotions": {}}}
    
    characters = cached_analysis.get("characters", [])
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
    if cached_analysis is None:
        return {"readability_score": 0}
    return {"readability_score": cached_analysis.get("readability_score", 0)}

@app.get("/narrative_direction")
def get_narrative_direction():
    if cached_analysis is None:
        return {"narrative_direction": "No analysis available"}
    return {"narrative_direction": cached_analysis.get("narrative_direction", "No analysis available")}

@app.get("/poetic_devices")
def get_poetic_devices():
    if cached_analysis is None:
        return {"poetic_devices": {}}
    return {"poetic_devices": cached_analysis.get("poetic_devices", {})}

#uvicorn scene_enhancer:app --reload
