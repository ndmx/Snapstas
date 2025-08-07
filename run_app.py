from ngrok import ngrok
import os
import subprocess
import sys
import time

# Your ngrok auth token
NGROK_AUTH_TOKEN = "30yNJYax8EUfMAIbMBYmcq21KI1_2zeJTSGTZCtJWHQSS5gTZ"

def setup_ngrok():
    # Configure ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Start ngrok tunnel
    try:
        # Connect to ngrok
        tunnel = ngrok.connect(8501)
        
        # Get the public URL
        public_url = str(tunnel)
        print(f"\nNgrok tunnel is running at: {public_url}")
        print("\nShare this URL to access your Streamlit app from anywhere!")
        return public_url
    except Exception as e:
        print(f"\nFailed to create ngrok tunnel: {str(e)}")
        return None

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
