import streamlit as st
import requests
import zipfile
import io

def download_and_extract_zip(zip_url):
    try:
        response = requests.get(zip_url)
        response.raise_for_status()  # Check if the request was successful

        if 'application/zip' not in response.headers.get('Content-Type', ''):
            raise ValueError('The file is not a ZIP archive.')

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall('/tmp')
            file_list = z.namelist()
            return file_list
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return []
    except zipfile.BadZipFile:
        st.error("The file is not a valid ZIP archive.")
        return []
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

def main():
    st.title('ZIP File Extractor')

    zip_url = st.text_input('Enter the URL of the ZIP file:')

    if zip_url:
        file_list = download_and_extract_zip(zip_url)
        if file_list:
            st.write("Files in the ZIP archive:")
            for file_name in file_list:
                st.write(file_name)
        else:
            st.write("No files found.")

if __name__ == "__main__":
    main()
