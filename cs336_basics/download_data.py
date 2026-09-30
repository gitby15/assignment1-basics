from pathlib import Path
from huggingface_hub import hf_hub_download
workspace_folder = Path(__file__).parent.parent
output_dir = workspace_folder / "data"


def download_tinystoriesv2():
    path = hf_hub_download(
        repo_id="roneneldan/TinyStories",
        repo_type="dataset",
        filename="TinyStoriesV2-GPT4-train.txt",
        local_dir=output_dir,
    )
    return Path(path)


def download_owt():
    path = hf_hub_download(
        repo_id="stanford-cs336/owt-sample",
        repo_type="dataset",
        filename="owt_train.txt.gz",
        local_dir=output_dir,
    )
    return Path(path)

if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)
    _path = download_owt()
    print(_path)