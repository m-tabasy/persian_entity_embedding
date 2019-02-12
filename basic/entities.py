import configparser
import os
import re

config = configparser.ConfigParser()
file_path = os.path.dirname(__file__)
config_path = os.path.join(file_path, '../config.ini')
config.read(config_path)

fa_wiki_path = config['path']['wiki']
word2vec_path = config['path']['word2vec']

ent_max = int(config['limit']['entity'])

# -------- globals --------

unk_ent_name = 'UNK_E'
unk_ent_id = -1
unk_ent_wiki_id = -1
ent_size = 0  # will be updated!

name2wikiid, wikiid2name = {}, {}
id2wikiid, wikiid2id = {}, {}


def extract_doc_tag(line):
	global unk_ent_wiki_id, unk_ent_name

	doc_pattern = re.compile(r'<doc[^>]+id=\"([0-9]+)\"[^>]+title=\"([^>\"]+)\"[^>]*>', re.DOTALL | re.UNICODE)

	doc_tag = re.match(doc_pattern, line)

	if doc_tag:
		doc_id, doc_title = tuple(doc_tag.groups())
		wiki_id = int(doc_id)
		return wiki_id, doc_title

	return unk_ent_wiki_id, unk_ent_name


def gen_entity_id_maps(wiki_path):
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
	print(f'done! {doc_count} entities added!')


gen_entity_id_maps(fa_wiki_path)