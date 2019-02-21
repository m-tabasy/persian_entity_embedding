import os
import re
import time

from gensim.corpora import wikicorpus
from hazm import word_tokenize

from basic.words import get_stop_words, word2id
from basic.entities import name2wikiid
from urllib.parse import unquote
from basic import commons

config = commons.get_config()

force_generate = config.getboolean('param', 'regenerate')
fa_wiki_path = config['path']['wiki']
hyp_ctx_words_path = config['path']['hyp_ctx']

context_size = int(config['param']['hyp_ctx_len'])


def extract_hyperlinks(line, context_size=10):
	hyp_pattern = re.compile(r'<a[^>]*href=\"([^\">]+)\"[^>]*>([^>]+)</a>', re.DOTALL | re.UNICODE)

	clean_text = wikicorpus.filter_wiki(line)
	hyp_matches = re.finditer(hyp_pattern, line)

	mentions = {}

	for link in hyp_matches:
		hyperlink = {}

		hyperlink['name'] = unquote(link.groups()[0])   # wikipedia url id
		hyperlink['mention'] = link.groups()[1]

		start_index = 0
		if hyperlink['mention'] in mentions:     # avoid repeating context for similar mentions in one line
			start_index = mentions[hyperlink['mention']] + 1

		left_index = clean_text.find(hyperlink['mention'], start_index)
		right_index = left_index + len(hyperlink['mention'])

		mentions[hyperlink['mention']] = left_index

		left_words = word_tokenize(clean_text[:left_index])
		right_words = word_tokenize(clean_text[right_index:])

		stop_words = get_stop_words()
		valid_left_words = list(filter(lambda w: w in word2id, left_words))
		pure_left_words = list(filter(lambda w: w not in stop_words, valid_left_words))
		left_context = pure_left_words[-min(len(pure_left_words), context_size):]

		valid_right_words = list(filter(lambda w: w in word2id, right_words))
		pure_right_words = list(filter(lambda w: w not in stop_words, valid_right_words))
		right_context = pure_left_words[:min(len(pure_right_words), context_size)]

		left_context_ids = list(map(lambda w: str(word2id[w]), left_context))
		right_context_ids = list(map(lambda w: str(word2id[w]), right_context))

		hyperlink['left_context'] = '\t'.join(left_context_ids)
		hyperlink['right_context'] = '\t'.join(right_context_ids)

		yield hyperlink


def gen_wiki_hyp_ctx(wiki_path, output_path, context_size=10):

	print(f'-- generating hyperlink context data based on {wiki_path} corpus...', end=' ', flush=True)

	with open(output_path, 'w', encoding='utf8') as outf:
		with open(wiki_path, 'r',encoding='utf8') as inf:

			for line in inf:
				for hyp in extract_hyperlinks(line, context_size=context_size):

					if hyp['name'] in name2wikiid:   # link to valid entity
						outf.write(str(name2wikiid[hyp['name']]) + '\t')
						outf.write(hyp['name'] + '\t')
						outf.write(hyp['mention'] + '\t')
						outf.write('LEFT_CTX\t' + hyp['left_context'] + '\t')
						outf.write('RIGHT_CTX\t' + hyp['right_context'] + '\n')

	print('done!')


if __name__ == '__main__':
	# generate data and calculate time
	_start = time.time()
	if force_generate or not os.path.exists(hyp_ctx_words_path):
		gen_wiki_hyp_ctx(fa_wiki_path, hyp_ctx_words_path, context_size=context_size)
	_end = time.time()
	print(f'-- calculated time is {_end - _start} sec')
