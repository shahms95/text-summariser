from collections import Counter
import nltk
from math import log, sqrt

wordType = {'content' : 0, 'punctuation' : 1, 'null' : -1, 'stopword' : 1}

class Graph:
	def __init__(self):
		self.vertices = []
		self.edges = []

	def addVertex(self, vertex):
		self.vertices.append(vertex)

	def addEdge(self, edge):
		self.edges.append(edge)

	def getVerticesSortedByWeight(self):
		
		sortedVertices = {}

		for e in self.edges:
			if e.v1 not in sortedVertices:
				sortedVertices[e.v1] = e.w
			else:
				sortedVertices[e.v1] += e.w

			if e.v2 not in sortedVertices:
				sortedVertices[e.v2] = e.w	
			else:
				sortedVertices[e.v2] += e.w

		return sorted(sortedVertices.items(), key = lambda x : x[1], reverse = True)

class Vertex:
	def __init__(self):
		self.sentence = None

	def __init__(self, s):
		self.sentence = s

class Edge:
	def __init__(self):
		self.v1 = None
		self.v2 = None
		self.w = 0.0

	def __init__(self, v1, v2, w):
		self.v1 = v1
		self.v2 = v2
		self.w = w

	def updateWeight(self, w):
		self.w = w

class Word:
	def __init__(self):
		self.type = -1
		self.text = None

	def __init__(self, type_, text):
		self.type = type_
		self.text = text

class Sentence:
	def __init__(self):
		self.words = []
		self.sentence = ''
	
	def __init__(self, sentence, stopWords):
		self.sentence = sentence
		self.words = []
		
		words = nltk.word_tokenize(sentence)
		
		contentPunctuationChars = '.?!\n'
		specialPunctionChars = ' -,'
		punctuationChars = contentPunctuationChars + specialPunctionChars

		for w in words:
			flag = False
			for x in punctuationChars:
				if w.lower() == x:
					self.words.append(Word(wordType['punctuation'], w))
					flag = True
					break
			if not flag:
				for x in stopWords:
					if w.lower() == x:
						self.words.append(Word(wordType['stopword'], w))
						flag = True
						break
			if not flag:			
				self.words.append(Word(wordType['content'], w))

	# get rid of punctuation and symbols	
	def getSimplifiedSentence(self):
		simplifiedSentence = ''
		usefulfulWords = filter(lambda x : x.type == wordType['content'], self.words)
		l = len(usefulWords)
		for i in range(l):
			if i < l-1:
				simplifiedSentence += meaningfulWords[i] + ' '
			else:
				simplifiedSentence += meaningfulWords[i]
		return simplifiedSentence

	def getContentWords(self):
		return filter(lambda x : x.type == wordType['content'], self.words)

class Paragraph:
	def __init__(self):
		self.sentences = []
		self.paragraph = None

	def __init__(self, paragraph, stopWords):
		self.sentences = [Sentence(x, stopWords) for x in nltk.tokenize.sent_tokenize(paragraph.decode('utf-8'))]
		self.paragraph = paragraph

class Similarity:
	def findJaccardSimilarity(self, s1, s2):
		s1 = s1.getContentWords()
		s1 = [x.text.lower() for x in s1]
		s2 = s2.getContentWords()
		s2 = [x.text.lower() for x in s2]

		l1 = len(s1)
		l2 = len(s2)

		if l1 < 5 or l2 < 5:
			return 0.0

		word_set1 = set(s1)
		word_set2 = set(s2)

		common_words = word_set1.intersection(word_set2)
		union_words = word_set1.union(word_set2)

		return float(len(common_words))/float(len(union_words))

	def findDiceSimilarity(self, s1, s2):
		s1 = s1.getContentWords()
		s1 = [x.text.lower() for x in s1]
		s2 = s2.getContentWords()
		s2 = [x.text.lower() for x in s2]

		l1 = len(s1)
		l2 = len(s2)

		if l1 < 5 or l2 < 5:
			return 0.0
		
		word_set1 = set(s1)
		word_set2 = set(s2)

		common_words = word_set1.intersection(word_set2)

		return 2*float(len(common_words))/(l1 + l2)

	def findRadaMihalceaSimilarity(self, s1, s2):
		s1 = s1.getContentWords()
		s1 = [x.text.lower() for x in s1]
		s2 = s2.getContentWords()
		s2 = [x.text.lower() for x in s2]

		l1 = len(s1)
		l2 = len(s2)

		if l1 < 5 or l2 < 5:
			return 0.0
		
		word_set1 = set(s1)
		word_set2 = set(s2)

		common_words = word_set1.intersection(word_set2)

		return float(len(common_words))/(log(l1) + log(l2))

	def findCosineSimilarity(self, s1, s2):
		s1 = s1.getContentWords()
		s1 = [x.text.lower() for x in s1]
		s2 = s2.getContentWords()
		s2 = [x.text.lower() for x in s2]

		l1 = len(s1)
		l2 = len(s2)

		if l1 < 5 or l2 < 5:
			return 0.0
		
		pair1 = Counter(s1)
		pair2 = Counter(s2)
		
		intersection = set(pair1.keys()).intersection(set(pair2.keys()))
		if len(intersection) == 0:
			return 0.0

		numerator = sum([pair1[x] * pair2[x] for x in intersection])

		denominator1 = sum([pair1[x]**2 for x in pair1.keys()])
		denominator2 = sum([pair2[x]**2 for x in pair2.keys()])

		denominator = sqrt(denominator1) * sqrt(denominator2)

		return float(numerator)/float(denominator)

	def getWeightFunction(self, i):
		if i == 0:
			return self.findJaccardSimilarity
		elif i == 1:
			return self.findDiceSimilarity
		elif i == 2:	
			return self.findRadaMihalceaSimilarity
		elif i == 3:
			return self.findCosineSimilarity


class TextRank:
	def textRank(self, graph):
		vertices = graph.vertices
		edges = graph.edges

		n = len(vertices)
		

		old_S = {}
		new_S = {}
		edge_wts = {}

		for x in vertices:
			new_S[x] = 0.0
			old_S[x] = 0.0
			edge_wts[x] = dict()
			for y in vertices:
				edge_wts[x][y] = 0.0

		e_sum = {}

		for e in edges:
			old_S[e.v1] += e.w

			if e.v1 not in e_sum:
				e_sum[e.v1] = e.w
			else:
				e_sum[e.v1] += e.w
		
		for e in edges:
			if e_sum[e.v2] == 0.0:
				edge_wts[e.v1][e.v2] = 0.0
			else:
				edge_wts[e.v1][e.v2] = e.w/e_sum[e.v2]
			
			if e_sum[e.v1] == 0.0:
				edge_wts[e.v2][e.v1] = 0.0
			else:
				edge_wts[e.v2][e.v1] = e.w/e_sum[e.v1]

		err = 10000
		TOL = 1e-4
		d = 0.4
		count = 0
		while err > TOL:
			err = 0
			for u in vertices:
				temp = 0
				for v in vertices:
					if not u == v:
						temp += edge_wts[u][v]*old_S[v] 
				new_S[u] = 1 - d + d*temp
				err += abs(new_S[u] - old_S[u])
			old_S = dict(new_S)
			count = count + 1

		print count
		ret = []
		for x in old_S:
			ret.append((x, old_S[y]))

		return ret

class Reduce:
	
	def constructGraph(self, sentences, i):
		graph = Graph()
		weightFunction = Similarity().getWeightFunction(i)
		
		for s in sentences:
			graph.addVertex(Vertex(s))

		for u in graph.vertices:
			for v in graph.vertices:
				if not u == v:
					e = Edge(u, v, weightFunction(u.sentence, v.sentence))
					graph.addEdge(e)

		return graph

	def summarize(self, filename):
		stopWords = map(str, nltk.corpus.stopwords.words('english'))		
		input_text = open(filename).read()

		paragraphs = [Paragraph(x.replace('\n', ' '), stopWords) for x in input_text.split('\n\n') if not x == '']
		
		ordered_sentences = []
		sentences_class = []
		
		for p in paragraphs:
			for s in p.sentences:
				sentences_class.append(s)
				ordered_sentences.append(s.sentence)

		graph = self.constructGraph(sentences_class, 3)
		
		sortedSentences = TextRank().textRank(graph)
		# sortedSentences = graph.getVerticesSortedByWeight()
		final_summary = {}
		
		i = 0
		for ss in sortedSentences:
			if i < 4:
				index = sentences_class.index(ss[0].sentence)
				final_summary[index] = ss[0].sentence.sentence + '\n'
				i = i + 1
			else:
				break

		return ''.join(final_summary.values())

print Reduce().summarize('input3.txt')