import os
import time
from typing import cast

import requests

CLIENT_ID = cast(str, os.getenv("ADOBE_CLIENT_ID"))
if not CLIENT_ID:
    raise OSError("ADOBE_CLIENT_ID not found")

CLIENT_SECRET = cast(str, os.getenv("ADOBE_CLIENT_SECRET"))
if not CLIENT_SECRET:
    raise OSError("ADOBE_CLIENT_SECRET not found")

class PDFAdobeExtractor:
    def __init__(self):
        self._access_token = ""
        self._uploadUri = ""
        self._assetID = ""
        self._resource_status_url = ""
        self._download_uri = ""

    def get_text_from_local_pdf(self, path: str, url: str) -> str:
        self.url = url
        try:
            self._get_access_token()
        except Exception as e:
            print('Raise error: ', str(e))
            raise BrokenPipeError(e)
        self._get_uploadUri_and_assetID()
        self._upload(path)
        self._extract()
        self._poll_extraction()
        text = self._parse_extracted_pdf()
        return text

    def _get_access_token(self):
        try:
            payload = {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET
            }
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            response = requests.post("https://pdf-services.adobe.io/token", data=payload, headers=headers)
            if response.status_code == 200:
                json = response.json()
                self._access_token = json['access_token']
            else:
                print('Raise error in else')
                raise BrokenPipeError(f"Error {response.status_code} when getting access_token")
        except Exception as e:
            print('Raise error in getccesstoken')
            raise BrokenPipeError(e)
    
    def _get_uploadUri_and_assetID(self):
        try:
            payload = {
                "mediaType": "application/pdf"
            }
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": CLIENT_ID,
                "Authorization": f"Bearer {self._access_token}"
            }
            response = requests.post("https://pdf-services.adobe.io/assets", headers=headers, json=payload)
            if response.status_code == 200:
                json = response.json()
                self._uploadUri = json['uploadUri']
                self._assetID = json['assetID']
            else:
                raise BrokenPipeError(f"Error {response.status_code} when getting uploadUri and assetID")
        except Exception as e:
            raise BrokenPipeError(f"Error getting uploadUri and assetID for PDF extraction: {e}")

    def _upload(self, path: str):
        try:
            headers = {
                "Content-Type": "application/pdf"
            }
            with open(path, "rb") as file:
                response = requests.put(
                    self._uploadUri,
                    data=file,
                    headers=headers,
                    timeout=120
                )
                if response.status_code in (200, 204):
                    print("Uploaded PDF: ", self.url)
                else:
                    print(f"Error {response.status_code} when uploading PDF")
        except Exception as e:
            raise BrokenPipeError(f"Error uploading PDF: {e}")
    
    def _extract(self):
        try:
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": CLIENT_ID,
                "Authorization": f"Bearer {self._access_token}"
            }
            payload = {
                "assetID": self._assetID
            }
            response = requests.post(
                "https://pdf-services-ue1.adobe.io/operation/extractpdf",
                json=payload,
                headers=headers,
                timeout=120
            )
            if response.status_code == 201:
                location = response.headers.get('Location')
                if not location:
                    raise BrokenPipeError("No Location header in extraction operation response")
                self._resource_status_url = location
            else:
                raise BrokenPipeError(f"Error {response.status_code} when requesting extraction operation")
        except Exception as e:
            raise BrokenPipeError(f"Error requesting PDF extraction: {e}")

    def _poll_extraction(self):
        while True:
            headers = {
                "X-API-Key": CLIENT_ID,
                "Authorization": f"Bearer {self._access_token}"
            }
            try:
                response = requests.get(self._resource_status_url, headers=headers)
                if response.status_code == 200:
                    json = response.json()
                    status = json["status"]
                    if status == "done":
                        self._download_uri = json["content"]["downloadUri"]
                        break
                    elif status == "in progress":
                        print("PDF extraction in progress on Adobe's servers...")
                    elif status == "failed":
                        print("Failed extracting PDF")
                    else:
                        print(f"Unknown anwer from API: {status}")
                else:
                    print(f"Error {response.status_code} when extracting PDF")
            except Exception as e:
                raise BrokenPipeError(f"Error getting PDF extraction status: {e}")
            time.sleep(1)
    
    def _parse_extracted_pdf(self) -> str:
        try:
            response = requests.get(self._download_uri)
            if response.status_code == 200:
                json = response.json()
                elements = json["elements"]
                full_text = "\n".join([element["Text"] for element in elements if "Text" in element])
                return full_text
            else:
                raise BrokenPipeError(f"Error {response.status_code} when reading extracted PDF")
        except Exception as e:
            raise BrokenPipeError(f"Error reading PDF: {e}")
