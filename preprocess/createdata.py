import numpy as np
import pandas as pd
#from pybedtools import BedTool
import os
from Bio import SeqIO
import string
import argparse
import random
from random import seed
from random import randint
import altschulEriksonDinuclShuffle as di
import re

hg19_without_N = "PN_fasta/reference/hg19_rm_Ns.bed"
hg38_without_N = "PN_fasta/reference/hg38_rm_Ns.bed"
hg19_without_N_repeat = "PN_fasta/reference/hg19_rm_repeat_Ns.bed"
hg38_without_N_repeat = "PN_fasta/reference/hg38_rm_repeat_Ns.bed"
hg19_bed_file = "PN_fasta/reference/hg19.fa.bed"
hg38_bed_file = "PN_fasta/reference/hg38.fa.bed"
hg19_genome = "PN_fasta/reference/human.hg19.genome"
hg38_genome = "PN_fasta/reference/human.hg38.genome"
hg19_fasta = "PN_fasta/reference/hg19.fa"
hg38_fasta = "PN_fasta/reference/hg38.fa"


def reverse_complement(dna):

    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','N': 'N'}
    return ''.join([complement[base] for base in dna[::-1]])

def gen_positive_file(input_list,augment=True):

	length =len(input_list) 
	if augment:
		for i in range(length,10000):
			idx=randint(0,len(input_list)-1)
			input_list.append(input_list[idx])
	return input_list      

def gen_neg_dinc_file(input_list):

	negative_list = []
	for sequence in input_list:
		negative_seq = di.dinuclShuffle(sequence)
		while negative_seq in input_list:
			while negative_seq in negative_list:
				negative_seq = di.dinuclShuffle(sequence)
		negative_list.append(negative_seq)
	return negative_list

def load_fasta(input):

	sequence_list = []
	fasta_sequences = SeqIO.parse(open(input),'fasta')
	with open(input) as outfile:
		for fasta in fasta_sequences:
			name, sequence = fasta.id, str(fasta.seq)
			upper_sequence = sequence.upper()
			sequence_list.append(upper_sequence)
			sequence_list.append(reverse_complement(upper_sequence))

	return sequence_list





def rewrite_bed(filename, sequence_length):
	f = BedTool(filename)
	positive_peak_counter = 0
	kept_counter = 0
	with open('positive.bed', 'w') as g:
		for line in f:
			positive_peak_counter+=1
			mid_point = int(int((int(line[1]) + int(line[2]))) / 2)
			original_len = int(line[2]) - int(line[1])
			if mid_point - int(sequence_length / 2) < 0:
				pass
			else:
				if sequence_length % 2 == 0:
					line[1] = mid_point - int(sequence_length / 2)
					line [2] = mid_point + int(sequence_length / 2)
					g.write('\t'.join(line) + "\n") 
					kept_counter += 1
				else:
					line[1] = mid_point - int(sequence_length / 2)
					line [2] = mid_point + int(sequence_length / 2) + 1
					g.write('\t'.join(line) + "\n")
					kept_counter += 1

	return positive_peak_counter, kept_counter


# rewrite_bed("a.bed" , 1000)


def create_bed_and_fasta(hg =38):
	if os.path.exists('positive_fasta.fa') is False:
		if hg == 19:
			os.system('bedtools getfasta -fi %s -bed positive.bed -fo positive_fasta.fa' %(hg19_fasta))        #fasta file with positive peak sequences
			# print("19")
		else: 
			os.system('bedtools getfasta -fi %s -bed positive.bed -fo positive_fasta.fa' %(hg38_fasta)) 
			# print("38")   
	if os.path.exists('all_excluded.bed') is False:
		if hg ==19:
			os.system('bedtools subtract -a %s -b %s > N_regions.bed' %(hg19_bed_file, hg19_without_N))
			os.system('multiIntersectBed -i positive.bed N_regions.bed > all_excluded.bed') 
			os.system('bedtools sort -i all_excluded.bed > sorted_all_excluded.bed')
			os.system('bedtools merge -i sorted_all_excluded.bed > merged_all_excluded.bed')
			# print("19")
		else: 
			os.system('bedtools subtract -a %s -b %s > N_regions.bed' %(hg38_bed_file, hg38_without_N))
			os.system('multiIntersectBed -i positive.bed N_regions.bed > all_excluded.bed') 
			os.system('bedtools sort -i all_excluded.bed > sorted_all_excluded.bed')          
			os.system('bedtools merge -i sorted_all_excluded.bed > merged_all_excluded.bed')
			# print("38")

# create_bed_and_fasta(19)

def create_negative_bed( pos_num, ratio , sequence_length,hg=38):
	total_negative_samples = round(pos_num * float(ratio))
	# print(total_negative_samples)
	if os.path.exists('random_negative.bed') is False:
		if hg == 19:
			os.system('bedtools random -g %s -n %s -l %s > random_negative.bed' %( hg19_genome,total_negative_samples, sequence_length))        
			# print("19")
		else: 
			os.system('bedtools random -g %s -n %s -l %s > random_negative.bed' %( hg38_genome,total_negative_samples, sequence_length)) 
			# print("38")   

	if os.path.exists('negative.bed') is False:
		if hg ==19:
			os.system('shuffleBed -i random_negative.bed -g %s -excl merged_all_excluded.bed > negative.bed' %(hg19_genome))
			# print("19")
		else: 
			os.system('shuffleBed -i random_negative.bed -g %s -excl merged_all_excluded.bed > negative.bed' %(hg38_genome))
			# print("38")
	if os.path.exists('negative_fasta.fa') is False:
		if hg == 19:
			os.system('bedtools getfasta -fi %s -bed negative.bed -fo random_negative_data.fa' %(hg19_fasta))        #fasta file with positive peak sequences
			# print("19")
		else: 
			os.system('bedtools getfasta -fi %s -bed negative.bed -fo random_negative_data.fa'%(hg38_fasta)) 
			# print("38") 



def read_seq_file(filename):
	with open(filename) as f:
	    lines = [line.rstrip() for line in f]
	
	sequence_list=[]
	for idx in range(1,len(lines)):
		element=lines[idx].split()
		sequence_list.append(element[2])
	return sequence_list

def install_genome_fa(hg=38):
	if hg ==19:
		if os.path.exists('reference/hg19.fa') is False:
			print("Downloading hg19 fasta file.")
			os.system('wget -P PN_fasta/reference/ http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz')
			os.system('gzip -d PN_fasta/reference/hg19.fa.gz')


	else: 
		if os.path.exists('reference/hg38.fa') is False:
			print("Downloading hg38 fasta file.")
			os.system('wget -P PN_fasta/reference/ http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz')
			os.system('gzip -d PN_fasta/reference/hg38.fa.gz')




if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument("--filename",help="Enter .fasta file")
	parser.add_argument("--augment",help="Augment data with random sampling", default=True)
	parser.add_argument("--neg_type",help="Enter negative data type: dinucleotide or random",default="dinucleotide")
	args=parser.parse_args()

	m=re.findall(r'(\S+?)_',args.filename )
	data_Name=m[0]
	
	if args.filename.endswith(".seq"):
		sequence_list = read_seq_file(args.filename)
	else:
		sequence_list = load_fasta(args.filename)

	seq_len=(len(sequence_list[0]))
	if args.augment == True:
		augmented_sequence_list = gen_positive_file(sequence_list)
	else:
		augmented_sequence_list = gen_positive_file(sequence_list, augment=False)
	final_num_seq = len(augmented_sequence_list)
	with open('{}_positive_data.fa'.format(data_Name), 'w') as f:
    		for item in augmented_sequence_list:
        		f.write("%s\n" % item)

	if args.neg_type == 'dinucleotide':
		dinucleo_negative_list = gen_neg_dinc_file(augmented_sequence_list)
		with open('{}_dinucleotide_negative_data.fa'.format(data_Name), 'w') as f:
			for item in dinucleo_negative_list:
				f.write("%s\n" % item)

	else:
		pos_count,kept_count = rewrite_bed('PN_fasta/b.bed', int(seq_len))
		create_bed_and_fasta(hg = 38)
		create_negative_bed(int(pos_count), final_num_seq, int(seq_len))
		os.system('rm -r positive.bed')
		os.system('rm -r N_regions.bed')
		os.system('rm -r negative.bed')
		os.system('rm -r random_negative.bed')
		os.system('rm -r all_excluded.bed')
		os.system('rm -r merged_all_excluded.bed')
		os.system('rm -r sorted_all_excluded.bed')
		os.system('rm -r positive_fasta.fa')









