# hashtagseg

- the training samples as shared by @akshala, slight change to that
  - for ex: "German" - "BIIIII"
  - changed to : "German" - "BBBBBB"
- those samples are saved in conll format for ease in loading. fields are [id,form,tag]
- test.conll : doesnt mean that it is test file. temporary name for the file. 
- test.conll is referenced in loadDataset.py
- loadDataset.py : used to load the conll format file in HuggingFace's Datasets library format. for ease of training. 
- full_pipeline.ipynb : used for performing hashtag segmentation

