from basic.words import *
from basic.entities import *

config = configparser.ConfigParser()
file_path = os.path.dirname(__file__)
config_path = os.path.join(file_path, '../config.ini')
config.read(config_path)

fa_wiki_path = config['path']['wiki']
canonical_words_path = config['path']['canonical']

ent_max = int(config['limit']['entity'])


def gen_canonical_words(wiki_path, output_path, replace_word_by_id=True):

	global ent_max

	print(f'-- generating canonical words based on {wiki_path} corpus...', end=' ', flush=True)

	output_line = ''
	doc_count = 0

	with open(output_path, 'w', encoding='utf8') as outf:
		with open(wiki_path, 'r', encoding='utf8') as inf:
			for line in inf:
				wiki_id, ent_name = extract_doc_tag(line)

				if wiki_id >= 0:    # new doc (entity)
					if len(output_line) > 0:
						outf.write(output_line + '\n')
						doc_count += 1
						if doc_count == ent_max:
							break

					output_line = f'{wiki_id}\t{ent_name}\tCANONICAL_WORDS'

				else:
					clean_text = wikicorpus.filter_wiki(line)
					words = word_tokenize(clean_text)

					# removing stop words from positive words
					stop_words = get_stop_words()
					valid_words = list(filter(lambda w: w in word2id.keys(), words))
					pure_words = list(filter(lambda w: w not in stop_words, valid_words))
					pure_ids = list(map(lambda w: str(word2id[w]), pure_words))

					if len(pure_words) > 0:
						output_line += '\t' + '\t'.join(pure_ids)

		outf.write(output_line + '\n')
	print(f'done!')


if __name__ == '__main__':
	gen_canonical_words(fa_wiki_path, canonical_words_path)
