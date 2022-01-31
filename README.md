## HashSet - A Dataset For Hashtag Segmentation
Hashset is a new dataset consisiting on 1.9k manually annotated and 3.3M loosely supervised tweets for testing the efficiency of hashtag segmentation models. We compare State of The Art Hashtag Segmentation models on Hashset and other baseline datasets (STAN and BOUN). We compare and analyse the results across the datasets to argue that HashSet can act as a good benchmark for hashtag segmentation tasks. 
<br> <br>

## Directory Tree
```
  '|-- HashtagSegmentation',
  '    |-- Data Stats Notebooks', ## ipynb notebooks used to generate statistics for the datasets used in this study
  '    |   |-- Data Validation.ipynb',  
  '    |   |-- Final Data Statistics.ipynb',
  '    |-- ModelPredictions',
  '    |   |-- distant-sampled-lowercase_hashformers_output.csv',
  '    |   |-- distant-sampled_hashformers_output.csv',
  '    |   |-- hashformer_analysis.ipynb',
  '    |   |-- maddela_analysis.ipynb',
  '    |   |-- stan-large_hashformers_output.csv',
  '    |   |-- stan-small_hashformers_output.csv',
  '    |   |-- Hashformers',
  '    |       |-- HashSet-Manual_hashformers_output.csv',
  '    |       |-- boun_hashformers_output.csv',
  '    |       |-- stan-dev_hashformers_output.csv',
  '    |-- datasets', ## Datasets used to compare model performances along with HashSet. 
  '        |-- boun-celebi-et-al.csv',  ## dataset used for comparison. Proposed by 
  '        |-- stan-dev-celebi-etal.csv',
  '        |-- stan-large-maddela_et_al_dev.pkl',
  '        |-- stan-large-maddela_et_al_test.pkl', 
  '        |-- stan-large-maddela_et_al_train.pkl',
  '        |-- stan-small-bansal_et_al.pkl', 
  '        |-- hashset',
  '            |-- HashSet-Distant-sampled.csv',
  '            |-- HashSet-Distant.csv',
  '            |-- HashSet-Manual.csv',
  ''
```
## HashSet
1. HashSet Manual: contains 1.9k manually annotated hashtags. Each row consists of the *hashtag*, *segmented hashtag* ,*named entity annotations*, a list storing whether the hashtag contains mix of hindi and english tokens and/or contains non-english tokens.
2. HashSet Distant: 3.3M loosely collected camel cased hashtags containing *hashtag* and their *segmentation*  



## Model Predictions
*Add content here* 
[Paper](https://arxiv.org/pdf/2201.06741.pdf)
