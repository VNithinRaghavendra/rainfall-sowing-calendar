import os
import sys
import http.server
import socketserver
import webbrowser

PORT = 8000
DIRECTORY = "dashboard"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from the dashboard directory
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    # Verify directory exists
    if not os.path.exists(DIRECTORY):
        print(f"Error: Directory '{DIRECTORY}' does not exist.")
        sys.exit(1)
        
    # Verify data.json exists inside the dashboard directory
    data_json_path = os.path.join(DIRECTORY, "data.json")
    if not os.path.exists(data_json_path):
        print(f"Error: Database file '{data_json_path}' not found. Please run export_dashboard_data.py first.")
        sys.exit(1)
        
    handler = CustomHTTPRequestHandler
    
    # Enable socket reuse to avoid "Address already in use" errors during quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print("===============================================================")
            print("         AGROCAST FORECASTING DASHBOARD DEPLOYED               ")
            print("===============================================================")
            print(f" Local Web Server: Running on http://localhost:{PORT} ")
            print(f" Serving Directory: '{DIRECTORY}/' ")
            print(" To terminate the deployment, press Ctrl+C in the terminal.    ")
            print("===============================================================")
            
            # Automatically open the dashboard in the default browser
            webbrowser.open(f"http://localhost:{PORT}")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDeployment server stopped. Thank you for using AgroCast!")
    except Exception as e:
        print(f"\nError starting server: {e}")

if __name__ == "__main__":
    start_server()
