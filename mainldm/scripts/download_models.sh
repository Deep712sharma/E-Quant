#!/bin/bash
wget -O mainldm/models/ldm/celeba256/celeba-256.zip https://ommer-lab.com/files/e-quant/celeba.zip
wget -O mainldm/models/ldm/ffhq256/ffhq-256.zip https://ommer-lab.com/files/e-quant/ffhq.zip
wget -O mainldm/models/ldm/lsun_churches256/lsun_churches-256.zip https://ommer-lab.com/files/e-quant/lsun_churches.zip
wget -O mainldm/models/ldm/lsun_beds256/lsun_beds-256.zip https://ommer-lab.com/files/e-quant/lsun_bedrooms.zip
wget -O mainldm/models/ldm/text2img256/model.zip https://ommer-lab.com/files/e-quant/text2img.zip
wget -O mainldm/models/ldm/cin256/model.zip https://ommer-lab.com/files/e-quant/cin.zip
wget -O mainldm/models/ldm/semantic_synthesis512/model.zip https://ommer-lab.com/files/e-quant/semantic_synthesis.zip
wget -O mainldm/models/ldm/semantic_synthesis256/model.zip https://ommer-lab.com/files/e-quant/semantic_synthesis256.zip
wget -O mainldm/models/ldm/bsr_sr/model.zip https://ommer-lab.com/files/e-quant/sr_bsr.zip
wget -O mainldm/models/ldm/layout2img-openimages256/model.zip https://ommer-lab.com/files/e-quant/layout2img_model.zip
wget -O mainldm/models/ldm/inpainting_big/model.zip https://ommer-lab.com/files/e-quant/inpainting_big.zip



cd mainldm/models/ldm/celeba256
unzip -o celeba-256.zip

cd mainldm/models/ldm/ffhq256
unzip -o ffhq-256.zip

cd mainldm/models/ldm/lsun_churches256
unzip -o lsun_churches-256.zip

cd mainldm/models/ldm/lsun_beds256
unzip -o lsun_beds-256.zip

cd mainldm/models/ldm/text2img256
unzip -o model.zip

cd mainldm/models/ldm/cin256
unzip -o model.zip

cd mainldm/models/ldm/semantic_synthesis512
unzip -o model.zip

cd mainldm/models/ldm/semantic_synthesis256
unzip -o model.zip

cd mainldm/models/ldm/bsr_sr
unzip -o model.zip

cd mainldm/models/ldm/layout2img-openimages256
unzip -o model.zip

cd mainldm/models/ldm/inpainting_big
unzip -o model.zip

cd ../..
