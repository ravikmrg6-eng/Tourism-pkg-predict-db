from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="customer_tourism_pkg_predict_proj_01/deployment",     # the local folder containing your files
    repo_id="ravikmrg6/Tourism-pkg-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
