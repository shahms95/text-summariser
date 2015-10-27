keyword_dict = {}
count = 0
ignore_list = []
length_threshold = 0

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
	return para
	# for x in para:
	# 	for y in pre_process_para(x):	
	# 		simplified.append(y)
	# return simplified

# Caculate the intersection between 2 sentences
def sentences_intersection(sent1, sent2):

    # split the sentence into words/tokens
    s1 = set(sent1.split(" "))
    s2 = set(sent2.split(" "))
    
    # If there is not intersection, just return 0
    if (len(s1) + len(s2)) == 0:
        return 0

    # We normalize the result by the average number of words
    return len(s1.intersection(s2)) / ((len(s1) + len(s2)) / 2)

def get_sentences_ranks(para):

    sentences = pre_process_para(para)

    # Calculate the intersection of every two sentences
    n = len(sentences)
    values = [[0 for x in xrange(n)] for x in xrange(n)]
    for i in range(0, n):
        for j in range(0, n):
            values[i][j] = sentences_intersection(sentences[i], sentences[j])
    
    # Build the sentences dictionary
    # The score of a sentences is the sum of all its intersection
    sentences_dic = {}
    for i in range(0, n):
        score = 0
        for j in range(0, n):
            if i == j:
                continue
            score += values[i][j]
        sentences_dic[sentences[i]] = score
    
    return sentences_dic

# Return the best sentence in a paragraph
def get_best_sentence(paragraph, sentences_dic):

    # Split the paragraph into sentences
    sentences = pre_process_para(paragraph)

    # Ignore short paragraphs
    if len(sentences) < 2:
        return ""

    # Get the best sentence according to the sentences dictionary
    best_sentence = ""
    max_value = 0
    for s in sentences:
        if sentences_dic[s] > max_value:
            max_value = sentences_dic[s]
            best_sentence = s

    return best_sentence

    # Build the summary
def get_summary(content, sentences_dic):

    # Split the content into paragraphs
    paragraphs = pre_process_input(content)

    # Add the title
    summary = []

    # Add the best sentence from each paragraph
    for p in paragraphs:
        sentence = get_best_sentence(p, sentences_dic).strip()
        if sentence:
            summary.append(sentence)

    return ("\n").join(summary)

# for x in open('ignore').read():
	# ignore_list.append(x)

f = open('inp')
# header = f.readline()
inp = f.read()
# processed = pre_process_input(inp)

sentences_dic = get_sentences_ranks(inp)
summary = get_summary(inp, sentences_dic)
print summary