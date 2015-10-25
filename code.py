import re

keyword_dict = {}
count = 0
ignore_list = []
length_threshold = 5

def pre_process_para(paragraph):
	sentences = paragraph.replace('etc.', 'etcetera')
	sentences = sentences.strip().split('.')
	sentences = [x.strip().replace('\n', ' ') for x in sentences]
	sentences = [x.replace('etcetera', 'etc.') for x in sentences]
	
	ret = []

	for sentence in sentences:
		y = sentence.split()
		for x in y:
			if x in keyword_dict:
				keyword_dict[x] = keyword_dict[x] + 1
			else:
				keyword_dict[x] = 1
			++count

		if len(y) >= length_threshold:
			ret.append(sentence)

	return ret

def pre_process_input(paras):
	simplified = []
	para = paras.strip().split('\n\n')
	for x in para:
		for y in pre_process_para(x):	
			simplified.append(y)
	return simplified

for x in open('ignore').read():
	ignore_list.append(x)

f = open('inp')
header = f.readline()
inp = f.read()





processed = pre_process_input(inp)

