import os
import subprocess
import time
from pyngrok import ngrok

def run_streamlit_with_ngrok():
    try:
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give Streamlit a moment to start up
        print("Starting Streamlit server...")
        time.sleep(3)
        
        # Set up ngrok tunnel to port 8501 (Streamlit's default port)
        public_url = ngrok.connect(8501)
        print(f"Streamlit app is running!")
        print(f"Local URL: http://localhost:8501")
        print(f"Public URL: {public_url}")
        
        try:
            # Keep the script running
            print("Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # Clean up on keyboard interrupt
            ngrok.disconnect(public_url)
            streamlit_process.terminate()
            print("Streamlit app has been stopped.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

run_streamlit_with_ngrok()