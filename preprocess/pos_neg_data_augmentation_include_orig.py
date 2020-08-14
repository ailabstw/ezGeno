
import argparse
# generate random integer values
from random import seed
from random import randint
import altschulEriksonDinuclShuffle as di
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()
    return args
"""
def gen_positive_file(inputfile,augment=True):
    #args = parse_args()
    seed(1)
    #initalize 
    seqlist=[]
    out=""
    #read inputfile
        
    #inputfile="./{}".format(args.input)
    print(inputfile)
    with open(inputfile, "r") as f:
        for line in f:
            line = line.strip()
            seqlist.append(line)
            out=out + line + "\n" 
        length =len(seqlist) 

    #write outputfile
    if augment:
        for i in range(length,10000):
            idx=randint(0,len(seqlist)-1)
            #print(idx)
            #print(seqlist[idx])
            out =out + seqlist[idx] +"\n"
    outputfile=inputfile.replace("positive","positive_augmentation_includeOrig")
    with open(outputfile, "w") as f:
        f.write(out)
    return outputfile

def gen_neg_dinc_file(inputfile):
    out =""
    N_counter = 0
    counter = 0
    with open(inputfile, "r") as f:
        #print(f.name)
        outFilename=f.name.replace("positive","negative_dinuclShuffle")
        for line in f:
    augment=True
            line =line.strip()
            if "N" in line:
                N_counter = N_counter +1
                counter = counter + 1
                continue
            seqList=[]
            while counter >0:
                seqList.append(line)
                negative_seq = di.dinuclShuffle(line)
                if negative_seq in seqList:
                    print("generate same sequence")
                    continue
                counter =counter -1
                out =out + negative_seq +"\n"

            negative_seq = di.dinuclShuffle(line)
            if negative_seq == line:
                print(line)
            #print(line)

            out =out + negative_seq +"\n"

    print(f.name+"\t"+str(N_counter))
    with open(outFilename, "w") as f:
        f.write(out)

def main():
    pos_file_name = gen_positive_file('SUZ12/SUZ12_positive_training.fa',augment=True)
    gen_neg_dinc_file(pos_file_name)    
    """
    pos_file_name = gen_positive_file('BDP1/BDP1_positive_training.fa')
    gen_negative_file(pos_file_name)
    pos_file_name = gen_positive_file('FAM48A/FAM48A_positive_training.fa')
    gen_negative_file(pos_file_name)
    pos_file_name = gen_positive_file('HDAC6/HDAC6_positive_training.fa')
    gen_negative_file(pos_file_name)
    pos_file_name = gen_positive_file('RXRA/RXRA_positive_training.fa')
    gen_negative_file(pos_file_name)
    """


if __name__ == "__main__":
    main()
