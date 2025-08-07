from ngrok import ngrok
import os
import subprocess
import sys
import time

# Your ngrok auth token
NGROK_AUTH_TOKEN = "30s8CoWhd1VmMkAx2zTfawQPHHS_685uXcb4KqXvbpvgfTkWJ"

def setup_ngrok():
    # Configure ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Start ngrok tunnel
    public_url = ngrok.connect(8502)  # Streamlit's default port
    print(f"\nNgrok tunnel is running at: {public_url}")
    print("\nShare this URL to access your Streamlit app from anywhere!")
    return public_url

def run_streamlit():
    # Run streamlit in a separate process
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    return subprocess.Popen(streamlit_cmd)

def main():
    try:
        # Start Streamlit
        print("Starting Streamlit app...")
        streamlit_process = run_streamlit()
        
        # Give Streamlit a moment to start
        time.sleep(3)
        
        # Setup ngrok
        print("\nSetting up ngrok tunnel...")
        public_url = setup_ngrok()
        
        print("\nPress Ctrl+C to stop the server")
        
        # Keep the script running
        streamlit_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        ngrok.kill()
        if 'streamlit_process' in locals():
            streamlit_process.terminate()

if __name__ == "__main__":
    main()
