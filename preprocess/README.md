# 1. Ezgeno preprocessing:
This preprocessing script helps to 
  
  (1) augment reverse-complement counterpart by user-defined parameter

  (1) augment positive data that contain less than 10,000 peaks into 10,000 data points using random sampling with replacement
  
  (2) generates negative instances of each positive peak using dinucleotide composition rules.
  
  (3) finally we will combine posistive data and negative data( eg:NFE2_training.sequence) for ezgeno input
## Usage
```bash
usage: createdata.py  [-h help] 
                  [--filename FILENAME]                  
                  [--augment AUGMENT] 
                  [--neg_type NEG_TYPE]
                  [--reverse REVERSE] 
                  [--outputprefix OUTPUTPREFIX]
                  
Required arguments:
  --filename    
                        input file name,file type can be .sequence (like deepbind input file format) or .fa
                        [Type: String]  
  --augment     
                        Augment data with random sampling. Recommended when data points are less than 10,000.
                        [Type: bool,default: False]  
  --neg_type
                        Select a type of negative data to train your model with or specify negative file  ex: "dinucleotide".
                        [Type: String,default:dinucleotide] 
  --reverse                 
                        Augment reverse-complement data .
                        [Type: bool,default: False]
  --outputprefix        
                        Add output prefix into filename.
                        [Type: String] 
```
## Example:

```python
 python3 createdata.py --filename JUND_HepG2_JunD_HudsonAlpha_AC.seq --reverse True --augment True
 ```
 
# 2. AcEnhancer preprocessing:
 Example dataset:
 you can download sample set from [here](https://drive.google.com/file/d/1qLk48r1tbmfhXsEiQhhz9kpYwuoVvJEQ/view?usp=sharing )
