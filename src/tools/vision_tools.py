import os
from PIL import Image
from google import genai
from src.tools.file_tools import PROJECT_ROOT

def analyze_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """Analyzes an image using Google Gemini Vision. Use this tool when the user asks about a photo."""
    try:
        target_path = (PROJECT_ROOT / image_path).resolve()
        if not target_path.is_relative_to(PROJECT_ROOT):
            return "Error: Path traversal detected. Access denied."
            
        if not target_path.exists():
            return f"Error: Image not found at {target_path}"
            
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY is not set in .env"
            
        client = genai.Client(api_key=api_key)
        img = Image.open(target_path)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img]
        )
        return f"Gemini Vision Report:\n{response.text}"
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"