import sys
from math import log

def print_help():
	sys.stderr.write('Usage: python calculate_idf.py corpus output_file\n')
	sys.stderr.write('corpus - each lemma in a row with an empty row as document separator\n')
	sys.exit(1)

if __name__=='__main__':
	idf={}
	if len(sys.argv)!=3:
		print_help()
	num_of_docs=0.0
	document_lemmata=set()
	output_file=open(sys.argv[2],'w')
	for line in open(sys.argv[1]):
		line=line.strip()
		if line=='':
			for lemma in document_lemmata:
				idf[lemma]=idf.get(lemma,0)+1
			num_of_docs+=1
			document_lemmata=set()
		else:
			document_lemmata.add(line)
	for lemma,freq in sorted(idf.items(),reverse=True,key=lambda x:x[1]):
		output_file.write(lemma+'\t'+str(round(log(num_of_docs/freq,2),4))+'\n')
