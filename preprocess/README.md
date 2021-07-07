# 1. Ezgeno preprocessing:
This preprocessing script helps to 

  (1) augment positive data that contain less than 10,000 peaks into 10,000 data points using random sampling with replacement
  
  (2) generates negative instances of each positive peak using dinucleotide composition rules.
  
  (3) finally we will get posistive data(eg:JUND_positive_data.fa) and negative data( eg:JUND_dinucleotide_negative_data.fa) for ezgeno input
## Usage
```bash
usage: createdata.py  [-h help] 
                  [--filename FILENAME]                  
                  [--augment AUGMENT] 
                  [--neg_type NEG_TYPE]
                  
Required arguments:
  --filename    
                        input file name,file type can be .seq (like deepbind input file format) or .fa
                        [Type: String]  
  --augment     
                        Augment data with random sampling. Recommended when data points are less than 10,000.
                        [Type: bool,default: True]  
  --neg_type
                        Select a type of negative data to train your model with. ex: "dinucleotide".
                        [Type: String,default:dinucleotide] 
```
## Example:

```python
 python3 createdata.py --filename JUND_HepG2_JunD_HudsonAlpha_AC.seq
 ```
 
# 2. AcEnhancer preprocessing:
 Example dataset:
 you can download sample set from [here](https://drive.google.com/file/d/1qLk48r1tbmfhXsEiQhhz9kpYwuoVvJEQ/view?usp=sharing )
