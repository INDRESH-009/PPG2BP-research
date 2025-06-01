import requests

def download_vitaldb_metadata():
    url = "https://api.vitaldb.net/cases"
    output_filename = "vitaldb_metadata.csv"

    try:
        print(f"Downloading metadata from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        # Decode the content using 'utf-8-sig' to handle BOM
        content = response.content.decode('utf-8-sig')

        # Save the content to a CSV file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Metadata saved to {output_filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading metadata: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_vitaldb_metadata()
