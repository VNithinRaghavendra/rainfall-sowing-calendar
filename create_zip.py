import os
import zipfile

def compress_project():
    zip_name = "rainfall_sowing_calendar_project.zip"
    
    root_files = [
        "requirements.txt",
        "download_data.py",
        "rainfall_prediction.py",
        "export_dashboard_data.py",
        "deploy_server.py",
        "build_notebook.py",
        "Rainfall_Prediction_Sowing_Calendar.ipynb",
        "PROJECT_REPORT.md",
        "sowing_calendar_advisory.txt",
        "README.md",
        "DEPLOYMENT_GUIDE.md",
        ".gitignore"
    ]
    
    # Directories and files to include recursively
    special_dirs = {
        "dashboard": ["index.html", "data.json"],
        "plots": [
            "01_subdivision_eda_trends.png",
            "02_stationarity_acf_pacf.png",
            "03_arima_predictions.png",
            "04_gbr_predictions.png",
            "05_model_comparison_metrics.png",
            "06_future_4week_forecast.png"
        ]
    }
    
    print(f"Creating project ZIP archive: '{zip_name}'...")
    
    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 1. Add root files
            for file in root_files:
                if os.path.exists(file):
                    zipf.write(file)
                    print(f"  + Added file: {file}")
                else:
                    print(f"  ! Warning: File '{file}' not found, skipping.")
                    
            # 2. Add special directories and their files
            for folder, files in special_dirs.items():
                if os.path.exists(folder):
                    for file in files:
                        file_path = os.path.join(folder, file)
                        if os.path.exists(file_path):
                            zipf.write(file_path)
                            print(f"  + Added file: {file_path}")
                        else:
                            print(f"  ! Warning: File '{file_path}' not found, skipping.")
                else:
                    print(f"  ! Warning: Folder '{folder}' not found, skipping.")
                    
        print("\n===============================================================")
        print("          ZIP ARCHIVE SUCCESSFULLY CONSTRUCTED                 ")
        print(f" Archive Name: {zip_name} ")
        print(f" File Size: {os.path.getsize(zip_name) / 1024 / 1024:.2f} MB ")
        print(f" Path: {os.path.abspath(zip_name)} ")
        print(" This ZIP archive is ready for GitHub uploads!                 ")
        print("===============================================================")
        return True
    except Exception as e:
        print(f"Error creating zip archive: {e}")
        return False

if __name__ == "__main__":
    compress_project()
