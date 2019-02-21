import os
import re
import pickle
import time

from numba import jit
from basic import commons

config = commons.get_config()

force_generate = config.getboolean('param', 'regenerate')
fa_wiki_path = config['path']['wiki']
word2vec_path = config['path']['word2vec']

ent_max = int(config['limit']['entity'])
entities_path = config['path']['entity']


# -------- globals --------

unk_ent_name = 'UNK_E'
unk_ent_id = -1
unk_ent_wiki_id = -1

ent_size = 0  # will be updated!

name2wikiid, wikiid2name = {}, {}
id2wikiid, wikiid2id = {}, {}


# -------- public functions --------

def is_special_page(title):
	
	if title.find('(ابهام‌زدایی)') > 0:
		# print(title)
		return True
	
	if title.find('فهرست') == 0:
		# print(title)
		return True
	
	return False


def extract_doc_tag(line):
	global unk_ent_wiki_id, unk_ent_name

	doc_pattern = re.compile(r'<doc[^>]+id=\"([0-9]+)\"[^>]+title=\"([^>\"]+)\"[^>]*>', re.DOTALL | re.UNICODE)

	doc_tag = re.match(doc_pattern, line)

	if doc_tag:
		doc_id, doc_title = tuple(doc_tag.groups())
		wiki_id = int(doc_id)
		if not is_special_page(doc_title):
			return wiki_id, doc_title

	return unk_ent_wiki_id, unk_ent_name


# -------- private by convention --------

def _gen_entity_id_maps(wiki_path):
	global ent_size, unk_ent_name, unk_ent_id, unk_ent_wiki_id, name2wikiid, wikiid2name, id2wikiid, wikiid2id

	print(f'-- looking for entity articles in {wiki_path} corpus...', end=' ', flush=True)

	doc_count = 0
	name2wikiid[unk_ent_name] = unk_ent_wiki_id
	wikiid2name[unk_ent_wiki_id] = unk_ent_name
	id2wikiid[unk_ent_id] = unk_ent_wiki_id
	wikiid2id[unk_ent_wiki_id] = unk_ent_id

	with open(wiki_path, 'r',encoding='utf8') as inf:

		for line in inf:

			wiki_id, doc_title = extract_doc_tag(line)

			if wiki_id >= 0:

				name2wikiid[doc_title] = wiki_id
				wikiid2name[wiki_id] = doc_title
				id2wikiid[doc_count] = wiki_id
				wikiid2id[wiki_id] = doc_count
				doc_count += 1

				if doc_count == ent_max:
					break

	ent_size = doc_count
	print(f'done!\n   {doc_count} entities added!')


def _init_entities():
	global ent_size, unk_ent_name, unk_ent_id, unk_ent_wiki_id, name2wikiid, wikiid2name, id2wikiid, wikiid2id
	
	if ent_size == 0:
		
		if force_generate or not os.path.exists(entities_path):
			
			if os.path.exists(entities_path):
				print('-- forcing generate entity id maps!')
			else:
				print('-- generating entity id maps because no file was found!')
				
			_gen_entity_id_maps(fa_wiki_path)
			
			with open(entities_path, 'wb') as outf:
				pickle.dump(ent_size, outf)
				pickle.dump(name2wikiid, outf)
				pickle.dump(wikiid2name, outf)
				pickle.dump(id2wikiid, outf)
				pickle.dump(wikiid2id, outf)
				
		else:
			
			print(f'-- loading entity id maps from {entities_path}...', end=' ', flush=True)
			
			with open(entities_path, 'rb') as inf:
				ent_size = pickle.load(inf)
				name2wikiid = pickle.load(inf)
				wikiid2name = pickle.load(inf)
				id2wikiid = pickle.load(inf)
				wikiid2id = pickle.load(inf)
			
			print(f'done!\n   {ent_size} entities loaded!')

# -------- test --------

if __name__ == '__main__':
	
	# check init time
	_start = time.time()
	_init_entities()
	_end = time.time()
	print(f'-- calculated time is {_end - _start} sec')
	
	# test
	# entity_names = list(name2wikiid.keys())
	# print('-- ' + '\n-- '.join(entity_names[:10]))

# -------- load when imported --------

else:
	_init_entities()
