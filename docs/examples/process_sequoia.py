"""Please, do:
pip install unlzw3 requests beautifulsoup4 numpy

DBSCAN paper using sequioa dataset:
https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf
"""
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import unlzw3
import numpy as np


# URL of the directory to download from
base_url = "http://s2k-ftp.cs.berkeley.edu/sequoia/benchmark/"

# Directory to save the downloaded files
download_dir = "sequoia_benchmark"
os.makedirs(download_dir, exist_ok=True)


def download_file(url, save_path):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

def scrape_and_download(url, save_dir):
    """Recursively scrape links from a directory listing and download files."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')

        for link in links:
            href = link.get('href')
            href_text = link.text.lower()
            if href not in ('../', '/') and href_text not in ('parent directory', 'name', 'last modified', 'size', 'description'):  # Exclude parent directory link
                full_url = urljoin(url, href)
                local_path = os.path.join(save_dir, href)

                if href.endswith('/'):
                    # It's a directory, create it locally and recurse
                    os.makedirs(local_path, exist_ok=True)
                    scrape_and_download(full_url, local_path)
                else:
                    # It's a file, download it
                    download_file(full_url, local_path)
    else:
        print(f"Failed to access {url}. Status code: {response.status_code}")


def sequoia_size(folder=download_dir):
    """Calculate the total size of the downloaded files."""
    total_size = 0
    nb_files = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
            nb_files += 1
    return (nb_files, total_size)


def decompress_and_read_z_file(file_path):
    """Decompress a .Z file and read the data into a NumPy array."""
    with open(file_path, 'rb') as compressed_file:
        compressed_data = compressed_file.read()

    # Decompress the data
    decompressed_data = unlzw3.unlzw(compressed_data)

    # Assuming the decompressed data is binary, load it into a NumPy array
    data = np.frombuffer(decompressed_data, dtype=np.uint8)  # Adjust dtype as needed
    return data


def load_sequoia_data(*, folder="graph", download_dir=download_dir):
    """Load the downloaded files into memory."""
    if folder not in ('graph', 'point', 'polygon', 'raster'):
        raise ValueError("folder must be one of 'graph', 'point', 'polygon', 'raster'")

    if folder in ['graph', 'point']:
        name = 'streams.Z' if folder == 'graph' else 'ca.Z'
        file_path = os.path.join(download_dir, folder, name)
        data = decompress_and_read_z_file(file_path)
        size_path = os.path.join(download_dir, folder, 'Sizes')
        with open(size_path, 'r') as f:
            sizes = f.read().split()

        if int(sizes[1]) != data.size:
            raise ValueError(f"Data shape {data.size} does not match expected shape {shape}")
        return data

    elif folder == 'polygon':
        size_path = os.path.join(download_dir, folder, 'Sizes')
        with open(size_path, 'r') as f:
            sizes = dict([line.split() for line in f])

        files = os.listdir(os.path.join(download_dir, folder))
        files = filter(lambda x: x.endswith('.Z'), files)
        all_data = []
        for file in files:
            file_path = os.path.join(download_dir, folder, file)
            data = decompress_and_read_z_file(file_path)

            if int(sizes.get(file[:-2], -1)) not in [data.size, -1]:
                raise ValueError(f"Data shape {data.size} does not match expected shape {file[:-2]}")
            all_data.append(data)
        return all_data

    elif folder == 'raster':
        size_path = os.path.join(download_dir, folder, 'Sizes')
        with open(size_path, 'r') as f:
            sizes = f.read().split()
        shape = int(sizes[0])

        files = os.listdir(os.path.join(download_dir, folder))
        files = filter(lambda x: x.endswith('.Z'), files)
        all_data = []
        for file in files:
            file_path = os.path.join(download_dir, folder, file)
            data = decompress_and_read_z_file(file_path)
            if shape != data.size:
                raise ValueError(f"Data shape {data.size} does not match expected shape {shape}")
            all_data.append(data)

        return np.array(all_data)


if __name__ == "__main__":
    # Check if the download directory is empty
    db_size = sequoia_size()
    if db_size[0] != 174:
        # Start the download process
        scrape_and_download(base_url, download_dir)

    print("Loading Sequoia data named Graph...")
    data1 = load_sequoia_data(folder="graph")
    print("Loading Sequoia data named Point...")
    data2 = load_sequoia_data(folder="point")
    print("Loading Sequoia data named Polygon...")
    data3 = load_sequoia_data(folder="polygon")
    # print("Loading Sequoia data named Raster...")
    # data4 = load_sequoia_data(folder="raster")


    t = 1
