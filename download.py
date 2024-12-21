import yt_dlp

url = "https://seed131.bitchute.com/Hi3MYpEer3nT/FIKpsBXxtqZD.mp4"


def download_mp3(url):
    id = url.split("/")[-1].split(".")[0]
    path = f"{id}.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "outtmpl": path,
    }
    print(f"Downloading video from {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
        return path


def download_mp4(url):
    id = url.split("/")[-1].split(".")[0]
    path = f"{id}.mp4"

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": path,
    }

    print(f"Downloading video from {url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return path


download_mp3(url)
download_mp4(url)
