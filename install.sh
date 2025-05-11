#!/usr/bin/env bash

# create the directories
mkdir -p data/images
mkdir -p data/labels

# images in a dir
wget -O data/images.tar.gz http://isis-data.science.uva.nl/jvgemert/images.tar.gz

# OCR labels in a dir
wget -O data/features.tar.gz http://isis-data.science.uva.nl/jvgemert/features.tar.gz

# json file with the annotations
wget -O data/annotations.zip https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/ConText-WordAnnotation.zip

## Extraction of the files

#tar -xzf data/images.tar.gz -C data/images --strip-components=1
#tar -xzf data/features.tar.gz -C data/labels --strip-components=1
#unzip -j data/annotations.zip "annotations.json" -d data/

# remove the compressed files
#rm data/images.tar.gz
#rm data/features.tar.gz
#rm data/annotations.zip
