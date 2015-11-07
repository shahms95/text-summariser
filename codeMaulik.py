# -*- coding: utf-8 -*-
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict,Counter
from string import punctuation
from heapq import nlargest

sentence_limit = 5
def remove_short_sentences(paragraph):
	sentences = []
	paragraph = paragraph.decode('utf-8')
	sents = sent_tokenize(paragraph)
	for s in sents:
		s_words = s.split()		
		if len(s_words) > sentence_limit:
			sentences.append(s)
	return sentences

def merge_dicts(dict1 , dict2):
	result_dict = {}
	if(len(dict2) < len(dict1)):
		temp_dict = dict1
		dict1 = dict2
		dict2 = temp_dict
	for w in dict1:
		if w in dict2:
			result_dict[w] = dict1[w]+dict2[w]
		else:
			result_dict[w] = dict1[w]
	for w in dict2:
		if w not in result_dict:
			result_dict[w] = dict2[w]
	return result_dict

def most_frequent_words(sentences):
	word_count = {}
	for sentence in sentences:
		sentence = sentence.strip()
		words = word_tokenize(sentence)
		word_count1 = Counter(words)
		word_count = merge_dicts(word_count, word_count1)
	return word_count

def main():
	content = """
	A Short sentence. This should go.repeat repeat repeat repeat repeat repeat. This should go too. However, This sentence should stay around. 
	repeat repeat repeat repeat repeat repeat repeat repeat.
    Abstract— Text Summarization is condensing the source text into a shorter version preserving its information content and overall meaning. It is very difficult for human beings to manually summarize large documents of text. Text Summarization methods can be classified into extractive and abstractive summarization.
    """
	# """
 #    An extractive summarization method consists of selecting important sentences, paragraphs etc. from the original document and concatenating them into shorter form. The importance of sentences is decided based on statistical and linguistic features of sentences. An abstractive summarization method consists of understanding the original text and re-telling it in fewer words.

 #     It uses linguistic methods to examine and interpret the text and then to find the new concepts and expressions to best describe it by generating a new shorter text that conveys the most important information from the original text document. In This paper, a Survey of Text Summarization Extractive techniques has been presented.
 #    """
	sentences = remove_short_sentences(content)
	word_count = most_frequent_words(sentences)
	for w in word_count:
		print w + "/*/*/*/* count : " 
		print word_count[w]
	# stopwords.words()

if __name__ == '__main__':
    main()