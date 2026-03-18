wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin
gdown https://drive.google.com/uc?id=1Wi4V2HFss6omhg557FgbRiEBSFxJCPXZ
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip ./brainACTIV_subj1_checkpoints.zip -d ./checkpoints/ && rm ./brainACTIV_subj1_checkpoints.zip
gdown --folder https://drive.google.com/drive/folders/1_4rNJEhdklBkOt-JeNjrvOdSZaHmF-UE