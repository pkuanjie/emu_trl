from huggingface_hub import snapshot_download

snapshot_download(repo_id="BAAI/Emu", cache_dir="/mnt/repos/Emu_RL/emu2_ckpts")
