import sys
import codecs
import re

def calculate_g0_dice(ngram_frequency_distr,unigram_frequency_distr):
	result=[]
	for ngram in ngram_frequency_distr:
		if ngram_frequency_distr[ngram]>=min_frequency: #minimum frequency
			result.append((float(ngram_length)*ngram_frequency_distr[ngram]/sum([unigram_frequency_distr[e] for e in ngram]),ngram))
	return sorted(result,reverse=True)

def calculate_g0_mi(ngram_frequency_distr,unigram_frequency_distr):
	result=[]
	ngram_probability=relativize_distr(ngram_frequency_distr)
	unigram_probability=relativize_distr(unigram_frequency_distr)
	from math import log
	for ngram in ngram_probability:
		if ngram_frequency_distr[ngram]>=min_frequency: # minimum frequency
			#result.append((log(ngram_frequency_distr[ngram]*ngram_probability[ngram]/product([unigram_probability[e] for e in ngram])),ngram))
			result.append(((log(ngram_probability[ngram]/product([unigram_probability[e] for e in ngram]))),ngram))
	return sorted(result,reverse=True)

def calculate_g0_ll(ngram_frequency):
	result=[]
	from math import log
	observed_functions=[observed2,observed3,observed4]
	expected_functions=[expected2,expected3,expected4]
	for ngram in lemma_token_frequency_distr:
		if ngram_frequency[ngram]>=min_frequency:
			marginals=[ngram_frequency[e] for e in create_ngram_star(ngram)]
			observed=observed_functions[ngram_length-2](marginals)
			expected=expected_functions[ngram_length-2](observed)
			ll=0.0
			for o,e in zip(observed,expected):
				try:
					ll+=o*log(o/e)
				except:
					pass
			result.append((2*ll,ngram))
	return sorted(result,reverse=True)

def calculate_g0_chisq(ngram_frequency):
	result=[]
	observed_functions=[observed2,observed3,observed4]
	expected_functions=[expected2,expected3,expected4]
	for ngram in lemma_token_frequency_distr:
		if ngram_frequency[ngram]>=min_frequency:
			marginals=[ngram_frequency[e] for e in create_ngram_star(ngram)]
			observed=observed_functions[ngram_length-2](marginals)
			expected=expected_functions[ngram_length-2](observed)
			chisq=0.0
			for o,e in zip(observed,expected):
				try:
					chisq+=(o-e)**2/e
				except:
					pass
			result.append((chisq,ngram))
	return sorted(result,reverse=True)

def calculate_g0_tscore(ngram_frequency):
	from math import sqrt
	result=[]
	observed_functions=[observed2,observed3,observed4]
	expected_functions=[expected2,expected3,expected4]
	for ngram in lemma_token_frequency_distr:
		if ngram_frequency[ngram]>=min_frequency:
			marginals=[ngram_frequency[e] for e in create_ngram_star(ngram)]
			observed=observed_functions[ngram_length-2](marginals)
			expected=expected_functions[ngram_length-2](observed)
			tscore=float(observed[0]-expected[0])/sqrt(expected[0])
			result.append((tscore,ngram))
	return sorted(result,reverse=True)

def observed2(marginals):
	n_ii,n_ix,n_xi,n_xx=marginals
	n_oi = n_xi - n_ii
	n_io = n_ix - n_ii
	return (n_ii, n_io, n_oi, n_xx - n_ii - n_oi - n_io)

def expected2(observed):
	n_ii, n_io, n_oi, n_oo=observed
	n_xx = float(n_ii+n_oi+n_io+n_oo)
	e_ii=(n_ii+n_io)*(n_ii+n_oi)/n_xx
	e_io=(n_ii+n_io)*(n_io+n_oo)/n_xx
	e_oi=(n_oi+n_oo)*(n_ii+n_oi)/n_xx
	e_oo=(n_oi+n_oo)*(n_io+n_oo)/n_xx
	return (e_ii,e_io,e_oi,e_oo) 
	
def expected_nltk(contingency):
	n_xx=sum(contingency)
	for i in range(4): 
		yield (contingency[i] + contingency[i ^ 1]) * (contingency[i] + contingency[i ^ 2]) / float(n_xx)

def observed3(marginals):
	n_iii,n_iix,n_ixi,n_ixx,n_xii,n_xix,n_xxi,n_xxx=marginals
	n_oii = n_xii - n_iii
	n_ioi = n_ixi - n_iii 
	n_iio = n_iix - n_iii 
	n_ooi = n_xxi - n_iii - n_oii - n_ioi
	n_oio = n_xix - n_iii - n_oii - n_iio 
	n_ioo = n_ixx - n_iii - n_ioi - n_iio 
	n_ooo = n_xxx - n_iii - n_oii - n_ioi - n_iio - n_ooi - n_oio - n_ioo 
	return (n_iii, n_iio, n_ioi, n_oii, n_ioo, n_ooi, n_oio, n_ooo)

def expected3(observed):
	o_iii, o_iio, o_ioi, o_oii, o_ioo, o_ooi, o_oio, o_ooo=observed
	n_xxx=float(o_iii+o_iio+o_ioi+o_oii+o_ioo+o_ooi+o_oio+o_ooo)
	e_iii=(o_iii+o_oii)*(o_iii+o_ioi)*(o_iii+o_iio)/n_xxx
	e_iio=(o_iio+o_oio)*(o_iio+o_ioi)*(o_iio+o_iii)/n_xxx
	e_ioi=(o_ioi+o_ooi)*(o_ioi+o_iii)*(o_ioi+o_ioo)/n_xxx
	e_oii=(o_oii+o_iii)*(o_oii+o_ooi)*(o_oii+o_oio)/n_xxx
	e_ioo=(o_ioo+o_ooo)*(o_ioo+o_iio)*(o_ioo+o_ioi)/n_xxx
	e_ooi=(o_ooi+o_ioi)*(o_ooi+o_oii)*(o_ooi+o_ooo)/n_xxx
	e_oio=(o_oio+o_iio)*(o_oio+o_ooo)*(o_oio+o_oii)/n_xxx
	e_ooo=(o_ooo+o_ioo)*(o_ooo+o_oio)*(o_ooo+o_ooi)/n_xxx
	return (e_iii,e_iio,e_ioi,e_oii,e_ioo,e_ooi,e_oio,e_ooo)

def observed4(marginals):
	n_iiii,n_xiii,n_ixii,n_iixi,n_iiix,n_xxii,n_ixxi,n_iixx,n_xixi,n_xiix,n_ixix,n_xxxi,n_xxix,n_xixx,n_ixxx,n_xxxx=marginals
	o_iiii=n_iiii
	o_oiii=n_xiii-n_iiii
	o_ioii=n_ixii-n_iiii
	o_iioi=n_iixi-n_iiii
	o_iiio=n_iiix-n_iiii
	o_ooii=n_xxii-n_iiii-o_oiii-o_ioii
	o_iooi=n_ixxi-n_iiii-o_ioii-o_iioi
	o_iioo=n_iixx-n_iiii-o_iioi-o_iiio
	o_oioi=n_xixi-n_iiii-o_oiii-o_iioi
	o_ioio=n_ixix-n_iiii-o_ioii-o_iiio
	o_oiio=n_ixxi-n_iiii-o_ioii-o_iioi
	o_oooi=n_xxxi-n_iiii-o_oiii-o_ioii-o_iioi-o_ooii-o_oioi-o_iooi
	o_ooio=n_xxix-n_iiii-o_oiii-o_ioii-o_iiio-o_ooii-o_oiio-o_ioio
	o_oioo=n_xixx-n_iiii-o_oiii-o_iioi-o_iiio-o_oioi-o_iioo-o_oiio
	o_iooo=n_ixxx-n_iiii-o_ioii-o_iioi-o_iiio-o_iooi-o_iioo-o_ioio
	o_oooo=n_xxxx-n_iiii-o_oiii-o_ioii-o_iioi-o_iiio-o_ooii-o_iooi-o_iioo-o_oioi-o_ioio-o_oiio-o_oooi-o_ooio-o_oioo-o_iooo
	return (o_iiii,o_oiii,o_ioii,o_iioi,o_iiio,o_ooii,o_iooi,o_iioo,o_oioi,o_ioio,o_oiio,o_oooi,o_ooio,o_oioo,o_iooo,o_oooo)

def expected4(observed):
	o_iiii,o_oiii,o_ioii,o_iioi,o_iiio,o_ooii,o_iooi,o_iioo,o_oioi,o_ioio,o_oiio,o_oooi,o_ooio,o_oioo,o_iooo,o_oooo=observed
	n_xxxx=float(o_iiii+o_oiii+o_ioii+o_iioi+o_iiio+o_ooii+o_iooi+o_iioo+o_oioi+o_ioio+o_oiio+o_oooi+o_ooio+o_oioo+o_iooo+o_oooo)
	e_iiii=(o_iiii+o_oiii)*(o_iiii+o_ioii)*(o_iiii+o_iioi)*(o_iiii+o_iiio)/n_xxxx
	e_oiii=(o_oiii+o_iiii)*(o_oiii+o_ooii)*(o_oiii+o_oioi)*(o_oiii+o_oiio)/n_xxxx
	e_ioii=(o_ioii+o_ooii)*(o_ioii+o_iiii)*(o_ioii+o_iooi)*(o_ioii+o_ioio)/n_xxxx
	e_iioi=(o_iioi+o_oioi)*(o_iioi+o_iooi)*(o_iioi+o_iiii)*(o_iioi+o_iioo)/n_xxxx
	e_iiio=(o_iiio+o_oiio)*(o_iiio+o_ioio)*(o_iiio+o_iioo)*(o_iiio+o_iiii)/n_xxxx
	e_ooii=(o_ooii+o_ioii)*(o_ooii+o_oiii)*(o_ooii+o_oooi)*(o_ooii+o_ooio)/n_xxxx
	e_oioi=(o_oioi+o_iioi)*(o_oioi+o_oooi)*(o_oioi+o_oiii)*(o_oioi+o_oioo)/n_xxxx
	e_oiio=(o_oiio+o_iiio)*(o_oiio+o_ooio)*(o_oiio+o_oioo)*(o_oiio+o_oiii)/n_xxxx
	e_iooi=(o_iooi+o_oooi)*(o_iooi+o_iioi)*(o_iooi+o_ioii)*(o_iooi+o_iooo)/n_xxxx
	e_ioio=(o_ioio+o_ooio)*(o_ioio+o_iiio)*(o_ioio+o_iooo)*(o_ioio+o_ioii)/n_xxxx
	e_iioo=(o_iioo+o_oioo)*(o_iioo+o_iooo)*(o_iioo+o_iiio)*(o_iioo+o_iioi)/n_xxxx
	e_oooi=(o_oooi+o_iooi)*(o_oooi+o_oioi)*(o_oooi+o_ooii)*(o_oooi+o_oooo)/n_xxxx
	e_ooio=(o_ooio+o_ioio)*(o_ooio+o_oiio)*(o_ooio+o_oooo)*(o_ooio+o_ooii)/n_xxxx
	e_oioo=(o_oioo+o_iioo)*(o_oioo+o_oooo)*(o_oioo+o_oiio)*(o_oioo+o_oioi)/n_xxxx
	e_iooo=(o_iooo+o_oooo)*(o_iooo+o_iioo)*(o_iooo+o_ioio)*(o_iooo+o_iooi)/n_xxxx
	e_oooo=(o_oooo+o_iooo)*(o_oooo+o_oioo)*(o_oooo+o_ooio)*(o_oooo+o_oooi)/n_xxxx
	return (e_iiii,e_oiii,e_ioii,e_iioi,e_iiio,e_ooii,e_iooi,e_iioo,e_oioi,e_ioio,e_oiio,e_oooi,e_ooio,e_oioo,e_iooo,e_oooo)

def create_ngram_star(lemmas):
	from itertools import product
	ngrams=[]
	for logical_vector in product((True,False),repeat=ngram_length):
		logical_ngram=['*' for e in range(ngram_length)]
		for index,logical_value in enumerate(logical_vector):
			if logical_value:
				logical_ngram[index]=lemmas[index]
		ngrams.append(tuple(logical_ngram))
	return ngrams

product=lambda s: reduce(lambda x, y: x * y, s)

def avg_idf(ngram):
	max_idf=max(idf.values())
	
	avg_idf=0.0
	for lemma in ngram:
		avg_idf+=idf.get(lemma,max_idf)
	return avg_idf/len(ngram)

def calculate_tfidf(ngram_frequency_distr):
	result=[]
	ngram_probability=relativize_distr(ngram_frequency_distr)
	for ngram,tf in ngram_probability.iteritems():
		if ngram_frequency_distr[ngram]>=min_frequency:
			idf_avg=0.0
			result.append((tf*avg_idf(ngram),ngram))
	return sorted(result,reverse=True)

def relativize_distr(distr):
	copy_distr={}
	all=float(sum(distr.values()))
	for key,value in distr.iteritems():
		copy_distr[key]=distr[key]/all
	return copy_distr

min_frequency=5
optional_properties=set(['h','s','n','t','o','pos','prop','min','idf','seq','norm','terms'])
mandatory_properties=set(['i','p','m','l'])
definable_properties=optional_properties.union(mandatory_properties)
properties={}
ranking_methods={'dice':calculate_g0_dice,'mi':calculate_g0_mi,'ll':calculate_g0_ll,'chisq':calculate_g0_chisq,'tscore':calculate_g0_tscore,'tfidf':calculate_tfidf}

def parse_arguments():
	key=None
	for argument in sys.argv[1:]:
		if argument.startswith('-'):
			key=argument[1:]
			if key=='h':
				print_help()
			if key not in definable_properties:
				sys.stderr.write('Warning: argument identifier '+argument+' unknown\n')
			else:
				properties[key]=[]
		elif key is None:
			sys.stderr.write('Warning: argument '+argument+' without argument identifier'+'\n')
		elif key in definable_properties:
			properties[key].append(argument)
		else:
			sys.stderr.write('Error: unknown argument identifier '+key+'\n')
			print_help()

def print_help():
	sys.stderr.write('Syntax:\n')
	sys.stderr.write('python CollTerm.py -i input_file -p phrase_configuration_file -m ngram_ranking_method -l length_of_ngrams\n')
	sys.stderr.write('Arguments:\n')
	sys.stderr.write('-i\ttab separated input file\n')
	sys.stderr.write('-p\tphrase configuration file\n')
	sys.stderr.write('\t\t- symbols for defining stop word phrases:\n')
	sys.stderr.write('\t\t\t- !STOP - non-stop word\n')
	sys.stderr.write('\t\t\t- STOP - stop word\n')
	sys.stderr.write('\t\t\t- * - any word\n')
	sys.stderr.write('\t\t- bigram example: STOP	*\n')
	sys.stderr.write('-s\tstop word file\n')
	sys.stderr.write('-n\tmaximum number of top ranked terms\n')
	sys.stderr.write('-t\tthreshold (minimum score)\n')
	sys.stderr.write('-m\tn-gram ranking method\n')
	sys.stderr.write('\t\tdice - Dice coefficient\n')
	sys.stderr.write('\t\tmi - modified mutual information\n')
	sys.stderr.write('\t\tchisq - chi-square statistic\n')
	sys.stderr.write('\t\tll - log-likelihood ratio\n')
	sys.stderr.write('\t\ttscore - t-score\n')
	sys.stderr.write('\t\ttfidf - tf-idf score (with a mandatory idf file)\n')
	sys.stderr.write('-l\tlength of n-grams to be extracted (0-4, 0 for all length n-grams)\n')
	sys.stderr.write('-o\textracted term list output file (if not given, stdout used)\n')
	sys.stderr.write('-pos\tpositions of tokens, POS tags and lemmas (zero-based indices)\n')
	sys.stderr.write('-min\tminimum frequency of n-grams taken in consideration\n')
	sys.stderr.write('-prop\tproperty file\n')
	sys.stderr.write('-idf\tidf file (mandatory for tfidf, in case of other ranking methods a linear combination is computed)\n')
	sys.stderr.write('-seq\toutput n-grams as a sequence of 0 - lemmata n-grams, 1 - most frequent token n-grams, 2 - all token n-grams with their frequencies\n')
	sys.stderr.write('-norm\tnormalize output to a [0,x] range\n')
	sys.stderr.write('-terms\toutput terms as\n')
	sys.stderr.write('\t\t0 - terms and weights (+ frequency if "-seq 2")\n')
	sys.stderr.write('\t\t1 - terms only\n')
	sys.exit(1)

def check_properties():		
	global term_number
	term_number=None
	global threshold
	threshold=None
	global ranking_method
	global ngram_length
	global output_file
	global position
	global min_frequency
	global seq
	seq=0
	global norm
	norm=0
	global terms
	terms=0
	# -prop
	if 'prop' in properties:
		if properties['prop']==[]:
			sys.stderr.write('Error: no argument given for property file (-prop)\n')
			print_help()
		import re
		line_re=re.compile('(.+?)=(.+)')
		try:
			properties_file=open(properties['prop'][-1])
		except:
			sys.stderr.write('Error: can not open property file (-prop): '+properties['prop'][-1]+'\n')
			print_help()
		i=0
		for line in properties_file:
			i+=1
			if line.strip().startswith('#'):
				continue
			result=line_re.search(line)
			if result is None:
				sys.stderr.write('Error in property file line '+str(i)+'\n'+repr(line)+'\n') 
				print_help()
			key=result.group(1).strip()
			if '"' in line:
				values=re.findall(r'"(.+?)"',result.group(2).strip())
			else:
				values=re.split(r'\s+',result.group(2).strip())
			if key not in definable_properties:
				sys.stderr.write('Error: unknown argument identifier '+key+' (property file)\n')
				print_help()
			properties[key]=values
	missing_properties=mandatory_properties.difference(mandatory_properties.intersection(properties))
	if len(missing_properties)>0:
		sys.stderr.write('Error: these mandatory properties are missing: '+', '.join(missing_properties)+'\n')
		print_help()

	# -n
	if 'n' in properties:
		if properties['n']==[]:
			sys.stderr.write('Error: no argument given for number of terms (-n)\n')
			print_help()		
		try:
			term_number=int(properties['n'][-1])
		except:
			sys.stderr.write('Error: wrong argument for number of terms (-n): '+properties['n'][-1]+'\n')
			print_help()
	# -t
	if 't' in properties:
		if properties['t']==[]:
			sys.stderr.write('Error: no argument given for threshold (-t)\n')
			print_help()
		try:
			threshold=float(properties['t'][-1])
		except:
			sys.stderr.write('Error: wrong argument for threshold (-t): '+properties['t'][-1]+'\n')
			print_help()
	# -m
	if properties['m']==[]:
		sys.stderr.write('Error: no argument given for n-gram ranking method (-m)\n')
		print_help()
	ranking_method=properties['m'][-1]
	if ranking_method not in ranking_methods:
		sys.stderr.write('Error: wrong argument for n-gram ranking method (-m): '+ranking_method+'\n')
		print_help()
	# -l
	if properties['l']==[]:
		sys.stderr.write('Error: no argument given for length of n-grams (-l)\n')
		print_help()
	try:
		ngram_length=int(properties['l'][-1])
		if ngram_length<1 or ngram_length>4:
			raise Exception
	except:
		sys.stderr.write('Error: wrong argument for length of n-grams (-l): '+properties['l'][-1]+'\n')
		print_help()
	#if ranking_method=='tfidf' and ngram_length!=1:
	#	sys.stderr.write('Error: tfidf ranking method applicable only to unigrams (-l 1)\n')
	#	print_help()
	if ngram_length==1 and 'idf' not in properties:
		sys.stderr.write('Error: idf file has to be defined for length of n-grams 1 (-l 1)\n')
		print_help()
	# -o
	if 'o' in properties:
		if properties['o']==[]:
			sys.stderr.write('Error: no argument given output file (-o)\n')
			print_help()
		try:
			output_file=open(properties['o'][-1],'w')
		except:
			sys.stderr.write('Error: can not write to output file '+properties['o'][-1]+'\n')
			print_help()	
	else:
		output_file=sys.stdout
	# -pos
	if 'pos' in properties:
		if len(properties['pos'])<3:
			sys.stderr.write('Error: wrong number of arguments given for position (-pos)\n')
			print_help()
		try:
			position=[int(e) for e in properties['pos'][-3:]]
			if len(set(position))!=3:
				raise Exception
		except:
			sys.stderr.write('Error: wrong arguments given for position (-pos): '+', '.join(properties['pos'][-3:])+'\n')
			print_help()
	else:
		position=range(3)
	# -min
	if 'min' in properties:
		if properties['min']==[]:
			sys.stderr.write('Error: no argument given for minimum frequency (-min)\n')
			print_help()
		try:
			min_frequency=int(properties['min'][-1])
		except:
			sys.stderr.write('Error: wrong argument given for minimum frequency (-min): '+properties['min'][-1]+'\n')
			print_help()
	# -idf
	if 'idf' in properties:
		if properties['idf']==[]:
			sys.stderr.write('Error: no argument given for idf file (-idf)\n')
			print_help()
	# -seq
	if 'seq' in properties:
		if properties['seq']==[]:
			sys.stderr.write('Error: no argument given for sequence output (-seq)\n')
			print_help()
		if properties['seq'][-1]!='0' and properties['seq'][-1]!='1' and properties['seq'][-1]!='2':
			sys.stderr.write('Error: wrong argument given for sequence output (-seq): '+properties['seq'][-1]+'\n')
			print_help()			
		seq=int(properties['seq'][-1])
	# -norm
	if 'norm' in properties:
		if properties['norm']==[]:
			sys.stderr.write('Error: no argument given for normalization (-norm)\n')
			print_help()
		try:
			norm=float(properties['norm'][-1])
		except:
			sys.stderr.write('Error: wrong argument given for normalization (-norm): '+properties['norm'][-1]+'\n')
			print_help()
	# -terms
	if 'terms' in properties:
		if properties['terms']==[]:
			sys.stderr.write('Error: no argument given for term output (-terms)\n')
			print_help()
		if properties['terms'][-1]!='0' and properties['terms'][-1]!='1':
			sys.stderr.write('Error: wrong argument given for term output (-terms): '+properties['terms'][-1]+'\n')
			print_help()			
		terms=int(properties['terms'][-1])

def read_phrase_configuration_file():
	global valid_phrases
	valid_phrases=[]
	global stop_phrase
	stop_phrase=[]
	global ngram_length
	if properties['p']==[]:
		sys.stderr.write('Error: no argument given for phrase configuration file (-p)\n')
		print_help()
	try:
		phrase_configuration_file=open(properties['p'][-1],'r')
	except:
		sys.stderr.write('Error: can not read phrase configuration file '+properties['p'][-1]+'\n')
		print_help()
	i=0
	for line in phrase_configuration_file:
		i+=1
		if line.strip().startswith('#'):
			continue
		if line.strip()=='':
			continue
		valid_phrase=line.strip().split('\t')
		try:
			if ngram_length!=len(valid_phrase):
				continue
			if 'STOP' in line:
				if stop_phrase!=[]:
					raise Exception
				for stop in valid_phrase:
					if stop not in ('!STOP','STOP','*'):
						raise Exception
				stop_phrase=valid_phrase
				continue
			valid_phrase=[re.compile(e) for e in valid_phrase]
		except:
			sys.stderr.write('Error in phrase configuration file line '+str(i)+'\n'+repr(line)+'\n')
			print_help()
		valid_phrases.append(valid_phrase)
	
def read_stopword_file():
	global stopwords
	stopwords=set()
	if 's' not in properties:
		return
	if properties['s']==[]:
		sys.stderr.write('Error: no argument given for stop-word file (-s)\n')
		print_help()
	try:
		stopwords=set([e.strip() for e in codecs.open(properties['s'][-1],'r','utf-8')])
	except:
		sys.stderr.write('Error: can not read stop-word file '+properties['s'][-1]+'\n')
		print_help()

def read_idf_file():
	if 'idf' not in properties:
		return
	global idf
	idf={}
	if properties['idf']==[]:
		sys.stderr.write('Error: no argument given for idf file (-idf)\n')
		print_help()
	try:
		idf=dict([(a,float(b)) for a,b in [e.split('\t') for e in codecs.open(properties['idf'][-1],'r','utf-8')]])
	except:
		sys.stderr.write('Error: can not read idf file '+properties['idf'][-1]+'\n')
		print_help()

def read_input_text():
	enamex=0
	numex=0
	timex=0
	if properties['i']==[]:
		sys.stderr.write('Error: no argument given for input file (-i)\n')
		print_help()
	path=properties['i'][-1]
	try:
		input_file=codecs.open(path,'r','utf-8')
	except:
		sys.stderr.write('Error: can not read input file '+path+'\n')
		print_help()
	i=0
	ngram=[None for e in range(ngram_length)]
	global ngram_frequency_distr
	ngram_frequency_distr={}
	global unigram_frequency_distr
	unigram_frequency_distr={}
	global lemma_token_frequency_distr
	lemma_token_frequency_distr={}
	for line in input_file:
		if i==0:
			if line.startswith(u'\ufeff'):
				line=line[1:]
		i+=1
		split_line=line.strip().split('\t')
		if split_line==['']:
			continue
		if split_line[0][0]=='<':
			if split_line[0].startswith('<NUMEX'):
				numex+=1
			elif split_line[0].startswith('<ENAMEX'):
				enamex+=1
			elif split_line[0].startswith('<TIMEX'):
				timex+=1
			elif split_line[0].startswith('</NUMEX'):
				numex-=1
			elif split_line[0].startswith('</ENAMEX'):
				enamex-=1
			elif split_line[0].startswith('</TIMEX'):
				timex-=1			
			ngram=[None for e in range(ngram_length)]
			if enamex<0 or numex<0 or timex<0:
				sys.stderr.write('Error: closing bracket without open on line '+str(i)+'\n'+repr(line)+'\n')
				sys.exit(1)
			continue
		if enamex!=0 or numex!=0 or timex!=0:
			continue
		try:
			token,pos,lemma=split_line[position[0]],split_line[position[1]],split_line[position[2]]
			token=token.lower()
			lemma=lemma.lower()
		except:
			sys.stderr.write('Error in input file line '+str(i)+'\n'+repr(line)+'\n')
			print_help()
		# actual stuff
		if ranking_method in ('dice','mi'):
			unigram_frequency_distr[lemma]=unigram_frequency_distr.get(lemma,0)+1
		ngram=ngram[1:]
		ngram.append((token,pos,lemma))
		try:
			tokens=tuple([e[0] for e in ngram])
			poses=tuple([e[1] for e in ngram])
			lemmas=tuple([e[2] for e in ngram])
		except:
			continue
		if ranking_method in ('ll','chisq','tscore'):
				for star_ngram in create_ngram_star(lemmas):
					ngram_frequency_distr[star_ngram]=ngram_frequency_distr.get(star_ngram,0)+1
		if check_ngram_for_phrases(poses) and check_ngram_for_stops(tokens):
			if ranking_method in ('dice','mi','tfidf'):
				ngram_frequency_distr[lemmas]=ngram_frequency_distr.get(lemmas,0)+1
			if lemmas not in lemma_token_frequency_distr:
				lemma_token_frequency_distr[lemmas]={}
			lemma_token_frequency_distr[lemmas][tokens]=lemma_token_frequency_distr[lemmas].get(tokens,0)+1
	global results
	if ranking_method in ('dice','mi'):
		results=ranking_methods[ranking_method](ngram_frequency_distr,unigram_frequency_distr)
	elif ranking_method in ('ll','chisq','tscore','tfidf'):
		results=ranking_methods[ranking_method](ngram_frequency_distr)
	
def check_ngram_for_phrases(pos_ngram):
	for phrase in valid_phrases:
		valid=True
		for pos,regex in zip(pos_ngram,phrase):
			if pos is None:
				valid=False
				break
			if regex.match(pos) is None:
				valid=False
				break
		if valid:
			return True
	return False

def check_ngram_for_stops(token_ngram):
	if stop_phrase==[]:
		return True
	for token,stop in zip(token_ngram,stop_phrase):
		if stop=='STOP' and token not in stopwords:
			return False
		elif stop=='!STOP' and token in stopwords:
			return False
	return True

def normalize_result():
	global results
	global norm
	if norm==0:
		norm=1
	values=[a for a,b in results]
	min_value=min(values)
	max_value=max(values)
	range=float(max_value-min_value)
	for index,(value,ngram) in enumerate(results):
		try:
			results[index]=(norm*(value-min_value)/range,ngram)
		except ZeroDivisionError:
			results[index]=(norm,ngram)

def linearly_combine_with_idf():
	normalize_result()
	global results
	results=sorted([(value*0.5+avg_idf(ngram)*0.5, ngram) for value,ngram in results],reverse=True)

def output_result():
	i=0
	for value,ngram in results:
		if threshold is not None:
			if value<threshold:
				break
		if seq==0:
			output=' '.join(ngram).encode('utf-8')
			if terms==0:
				output+='\t'+str(round(value,2))
			output_file.write(output+'\n')
		elif seq==1:
			output=' '.join(sorted(lemma_token_frequency_distr[ngram].items(),key=lambda x:x[1])[-1][0]).encode('utf-8')
			if terms==0:
				output+='\t'+str(round(value,2))
			output_file.write(output+'\n')
		else:
			for token_sequence,frequency in lemma_token_frequency_distr[ngram].iteritems():
				output=' '.join(token_sequence).encode('utf-8')
				if terms==0:
					output+='\t'+str(frequency)+'\t'+str(round(value,2))
				output_file.write(output+'\n')
		i+=1
		if term_number is not None:
			if i>=term_number:
				break

if __name__=='__main__':
	parse_arguments()
	check_properties()
	read_phrase_configuration_file()
	read_stopword_file()
	read_idf_file()
	read_input_text()
	if ranking_method!='tfidf' and 'idf' in vars():
		linearly_combine_with_idf()
	if norm!=0:
		normalize_result()
	output_result()
