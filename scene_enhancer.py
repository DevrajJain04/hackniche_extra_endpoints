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
scene_reports = []

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


def analyze_script(script_content):
    """Analyzes the script and ensures JSON formatted output."""
    prompt = f"""
    Analyze the following scene and strictly return a well-formed JSON object with this structure:
    // keep the threshold for something to be considered an emotion or a description to be low
    ```json
    {{
        "scene_description": "",
        "sound_effects": [],
        "visual_cues": [],
        "characters": [
            {{
                "name": "",
                "emotion": "",
                "description": ""
            }}
        ],
        "readability_score": "",
        "sentiment": "",
        "poetic_devices": {{}},
        "narrative_direction": ""
    }}
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
    if not script.script_text.strip():
        raise HTTPException(status_code=400, detail="Script text cannot be empty.")
    
    analysis = analyze_script(script.script_text)
    scene_reports.append(analysis)  # Append new scene analysis
    
    return {"analysis": analysis, "scene_count": len(scene_reports)}

@app.get("/report")
def get_aggregated_report():
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
    if scene_reports[-1] is None:
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
    if scene_reports[-1] is None:
        return {"readability_score": 0}
    return {"readability_score": scene_reports[-1].get("readability_score", 0)}

@app.get("/narrative_direction")
def get_narrative_direction():
    if scene_reports[-1] is None:
        return {"narrative_direction": "No analysis available"}
    return {"narrative_direction": scene_reports[-1].get("narrative_direction", "No analysis available")}

@app.get("/poetic_devices")
def get_poetic_devices():
    if scene_reports[-1] is None:
        return {"poetic_devices": {}}
    return {"poetic_devices": scene_reports[-1].get("poetic_devices", {})}

#uvicorn scene_enhancer:app --reload
