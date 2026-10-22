# Data

Nothing in this folder except this file goes into Git (see `.gitignore`).

| Folder | Contents | How to get it | License |
|---|---|---|---|
| `synthetic/` | Customers, policies, claims history, gold claims, documents, voice scripts, red-team, fine-tuning data | `python generate_all.py` (deterministic: same seed, same data) | Yours |
| `public/cardd/` | CarDD car damage images (COCO format) | Licence form at cardd-ustc.github.io, emailed to the authors; or the Hugging Face mirror `mouen5/CarDD` converted with `training/yolo/prepare_cardd_hf.py` | Non-commercial research and education only; images belong to Flickr/Shutterstock owners; do not redistribute |
| `public/kaggle_vehicle_fraud/` | Vehicle insurance fraud table | kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection | Check the Kaggle licence tab |Oct 22, 2026
| `public/policy_references/` | Real auto policy PDFs, for reference only | Oklahoma Insurance Department policy forms page | Copyrighted: don't commit or republish | Oct 22, 2026
| `features/` | Spark fraud features | `python -m training.fraud.spark_features` | Yours |

To keep large data in Google Drive, set `CLAIMSENSE_DATA_DIR` to the Drive folder (in Colab: `/content/drive/MyDrive/claimsense/data`).

Record the download date of each public dataset here when you get it.
