import os
from huggingface_hub import HfApi

def main():
    hf_token = os.environ["HF_API_KEY"]
    HFAPI = HfApi(token=hf_token)
    HFAPI.restart_space(repo_id="DelightNews/the-delight-dispatch-demo", token=hf_token)

if __name__ == "__main__":
    main()
