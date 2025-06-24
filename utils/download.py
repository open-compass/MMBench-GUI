import os
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
import time
import hashlib
from vlmeval.smp import LMUDataRoot
from vlmeval.smp import get_logger

ROOT = LMUDataRoot()
logger = get_logger("MMBench-GUI")

FILES = [
    {
        "name": "MMBench-GUI-OfflineImages.tar",
        "md5": "8a34d69f0e9c0c450a48bfc1c90b65d5",
        "type": "tar",
    },
    {
        "name": "L1_annotations.json",
        "md5": "a9a4fd9eb1e4ae0355fee057bd55cbec",
        "type": "json",
    },
    {
        "name": "L2_annotations.json",
        "md5": "a2e042302ac389299873a857b7370a35",
        "type": "json",
    },
]


def download_benchmark_data(
    url,
    download_path,
    extract_dir,
    max_retries=3,
    chunk_size=8192,
    verify_ssl=True,
):
    logger.info("Start to download benchmark images and json files from our server.")
    download_dir = Path(download_path).parent
    download_dir.mkdir(parents=True, exist_ok=True)
    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    download_status = download_file(
        url, download_path, max_retries, chunk_size, verify_ssl
    )
    if not download_status:
        return False

    if Path(download_path).filepath.suffix.lower() in [
        ".tar.gz",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".tar.bz2",
        ".tar.xz",
    ]:
        if not extract_file(download_path, extract_dir):
            return False
    else:
        pass

    print(
        f"✅ Images and json files are successfully downloaded and extracted to: {extract_dir}"
    )
    return True


def download_file(url, filepath, max_retries=3, chunk_size=8192, verify_ssl=True):
    for attempt in range(max_retries):
        try:
            logger.info(
                f"📥 Begin to download (try {attempt + 1}/{max_retries}): {url}"
            )

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.head(
                url, headers=headers, verify=verify_ssl, timeout=30
            )
            total_size = int(response.headers.get("content-length", 0))

            resume_header = {}
            initial_pos = 0
            if os.path.exists(filepath):
                initial_pos = os.path.getsize(filepath)
                if initial_pos < total_size:
                    resume_header["Range"] = f"bytes={initial_pos}-"
                    logger.info(
                        f"🔄 Find unfinised downloaded files, continue to download from {initial_pos} byte..."
                    )
                elif initial_pos == total_size:
                    logger.info("✅ Files are downloaded before, skip to download...")
                    return True

            response = requests.get(
                url,
                headers={**headers, **resume_header},
                stream=True,
                verify=verify_ssl,
                timeout=30,
            )
            response.raise_for_status()

            mode = "ab" if initial_pos > 0 else "wb"
            with open(filepath, mode) as f, tqdm(
                desc="Downloading Status",
                total=total_size,
                initial=initial_pos,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info("✅ Finish download process!")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Download error (try {attempt + 1}/{max_retries}): {str(e)}"
            )
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.info(f"⏳ wait {wait_time} second and try again...")
                time.sleep(wait_time)
            else:
                logger.error("❌ reach the maximal retry times, Soooo sad!")
                return False
        except Exception as e:
            logger.error(f"❌ Unknown error: {str(e)}")
            return False

    return False


def extract_file(filepath, extract_dir):
    try:
        filepath = Path(filepath)
        extract_dir = Path(extract_dir)

        logger.info(f"📦 Begin to extract: {filepath.name}")

        if filepath.suffix.lower() == ".zip":
            extract_zip(filepath, extract_dir)
        elif filepath.suffix.lower() in [
            ".tar",
            ".tar.gz",
            ".tgz",
            ".tar.bz2",
            ".tar.xz",
        ]:
            extract_tar(filepath, extract_dir)
        elif filepath.suffix.lower() == ".gz" and not filepath.name.endswith(".tar.gz"):
            extract_gzip(filepath, extract_dir)
        else:
            logger.warning(f"⚠️ Unsupported format: {filepath.suffix}")
            logger.warning(
                "We can process: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz, .gz"
            )
            return False

        logger.info("✅ Finish extracting files.")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to extracting error: {str(e)}")
        return False


def extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()

        with tqdm(desc="Extracting status", total=len(file_list), unit="files") as pbar:
            for file_name in file_list:
                zip_ref.extract(file_name, extract_dir)
                pbar.update(1)


def extract_tar(tar_path, extract_dir):
    if tar_path.name.endswith(".tar.gz") or tar_path.name.endswith(".tgz"):
        mode = "r:gz"
    elif tar_path.name.endswith(".tar.bz2"):
        mode = "r:bz2"
    elif tar_path.name.endswith(".tar.xz"):
        mode = "r:xz"
    else:
        mode = "r"

    with tarfile.open(tar_path, mode) as tar_ref:
        members = tar_ref.getmembers()

        with tqdm(desc="Extracting status", total=len(members), unit="files") as pbar:
            for member in members:
                tar_ref.extract(member, extract_dir)
                pbar.update(1)


def extract_gzip(gz_path, extract_dir):
    output_filename = gz_path.stem
    output_path = Path(extract_dir) / output_filename

    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:

            f_in.seek(0, 2)
            total_size = f_in.tell()
            f_in.seek(0)

            with tqdm(
                desc="Extracting status",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    chunk = f_in.read(8192)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))


def calculate_file_hash(filepath, algorithm="md5"):
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


if __name__ == "__main__":
    # Example
    for file in FILES:
        url = f"https://huggingface.co/datasets/huiserwang/MMBench-GUI/resolve/main/{file['name']}"
        download_path = os.path.join(
            "/mnt/petrelfs/wangxuehui", "MMBench-GUI", f"{file['name']}"
        )
        extract_dir = os.path.join("/mnt/petrelfs/wangxuehui", "MMBench-GUI")

        success = download_benchmark_data(url, download_path, extract_dir)

        # optional： validate md5 of downloaded files
        if success:
            if os.path.exists(download_path):
                logger.info("Validate MD5...")
                file_hash = calculate_file_hash(download_path)
                if file_hash == file["md5"]:
                    logger.info(f"MD5: {file_hash} match!")
                else:
                    logger.error(f"MD5: {file_hash} unmatch! Please check it!")

            if file["type"] != "json":
                logger.info("Remove downloaded .tar file to save disk")
                os.remove(download_path)

            logger.info("🎉 All done!")
        else:
            logger.info("💥 Failed! Please check your network connection and path")
