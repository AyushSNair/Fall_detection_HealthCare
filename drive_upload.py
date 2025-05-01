from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
import os

def upload_to_drive(filepath):
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)

    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": os.path.basename(filepath),
        "parents": ["1AZ2iUH_vzVREMgyrPCNQe1jMSDvQojyh"]  # Change this to your Drive folder ID
    }
    media = MediaFileUpload(filepath, mimetype="video/mp4")

    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    file_id = uploaded_file.get("id")

    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()

    drive_link = f"https://drive.google.com/file/d/{file_id}/view"
    return drive_link
