import os

import dropbox
from dropbox import DropboxOAuth2FlowNoRedirect
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

################################################################################
## Connect to Dropbox

# Get the keys & access code from the environment variables
DbxAppKey= os.getenv('DROPBOX_APP_KEY')
DbxAppSecret= os.getenv('DROPBOX_APP_SECRET')
DbxAccessToken= os.getenv('DROPBOX_ACCESS_TOKEN')
DbxRefreshToken= os.getenv('DROPBOX_REFRESH_TOKEN')

# Create a Dropbox client instance
dbx = dropbox.Dropbox(
    app_key=DbxAppKey,
    app_secret=DbxAppSecret,
    oauth2_refresh_token=DbxRefreshToken
)

################################################################################
## File I/O

def DbxFIO(DbxPath, mode="read", type="json", data=None):
    if mode == "read":
        metadata, res = dbx.files_download(DbxPath)
        if type == "json":
            json_data = json.loads(res.content.decode('utf-8'))
            return json_data
        elif type == "txt":
            return res.content.decode('utf-8')
        else:
            raise ValueError(f"Invalid type: {type}")
    elif mode == "write":
        if data is None:
            raise ValueError("Data is required for writing")
        if type == "json":
            data = json.dumps(data)
        elif type == "txt":
            data = str(data)
        else:
            raise ValueError(f"Invalid type: {type}")

        # Convert string data to bytes before uploading
        data = data.encode('utf-8')

        try:
            # Upload the file
            metadata = dbx.files_upload(data, DbxPath, mode=dropbox.files.WriteMode("overwrite"))
            print("Uploaded file:", metadata.name)
        except dropbox.exceptions.ApiError as err:
            print("API error:", err)























################################################################################
## Download the TensorFile from Dropbox

# try:
#     # Download the file
#     metadata, res = dbx.files_download(DbxPath)
#     # Decode and parse JSON content
#     json_data = json.loads(res.content.decode('utf-8'))
#     print("JSON content:", json_data)
# except dropbox.exceptions.ApiError as err:
#     print("API error:", err)

# ################################################################################
# ## Upload a file to Dropbox

# # Open the local file and read its contents
# with open('./Temp/EmoTensor-0-0.etsc', 'rb') as f:
#     file_data = f.read()

# # Define the destination path in your Dropbox (e.g., root folder)
# destination_path = '/EmoTensor/data/EmoTensor-0-0.etsc'

# try:
#     # Upload the file. WriteMode("add") creates the file only if it doesn't exist.
#     metadata = dbx.files_upload(file_data, destination_path, mode=dropbox.files.WriteMode("add"))
#     print("Uploaded file:", metadata.name)
# except dropbox.exceptions.ApiError as err:
#     print("API error:", err)

# Use this function to manually authenticate the Dropbox App and get the access token and refresh token
def DbxAuth():
    def authenticate():
        auth_flow = DropboxOAuth2FlowNoRedirect(
            DbxAppKey, 
            DbxAppSecret, 
            token_access_type='offline'
        )

        authorize_url = auth_flow.start()
        print("1. Go to: " + authorize_url)
        print("2. Click 'Allow' (you might have to log in first).")
        print("3. Copy the authorization code.")

        auth_code = input("Enter the authorization code here: ").strip()

        try:
            oauth_result = auth_flow.finish(auth_code)
        except Exception as e:
            print('Error: %s' % (e,))
            return

        print("\nAuthorization successful!")
        print("Access Token:", oauth_result.access_token)
        print("Refresh Token:", oauth_result.refresh_token)
        print("Account ID:", oauth_result.account_id)

        # Optionally, save tokens to environment variables or a file
        os.environ['DROPBOX_ACCESS_TOKEN'] = oauth_result.access_token
        os.environ['DROPBOX_REFRESH_TOKEN'] = oauth_result.refresh_token
        os.environ['DROPBOX_ACCOUNT_ID'] = oauth_result.account_id

        print("\nTokens have been set as environment variables:")
        print("DROPBOX_ACCESS_TOKEN")
        print("DROPBOX_REFRESH_TOKEN")
        print("DROPBOX_ACCOUNT_ID")

    if __name__ == "__main__":
        authenticate()
