#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import sys
from os import listdir
from os.path import isfile, join
import csv
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

tags_dict = {}

tag_files = [ f for f in listdir('./tags/') if isfile(join('./tags/',f)) ]
for tag_file in tag_files:
    if tag_file[-4:] == '.txt':
    	print(tag_file)
    	with open(('./tags/' + tag_file)) as tag_file_file:
    		for line in tag_file_file:
    			a = []
    			for token in line.strip().split():
    				a.append(morph.parse(token.lower())[0].normal_form)
	    		tags_dict[' '.join(a)] = tag_file[:-4]
print(len(tags_dict))
facets = ['acm', 'action', 'agregator', 'amn', 'city', 'contacts', 'date', 'geo', 'poi', 'price', 'stars', 'time', 'wh']

with open('data.csv/data_lem.csv') as data_file:
	reader = csv.DictReader(data_file, doublequote=False, escapechar='\\')
	with open('data.csv/data_lem_tags.csv', 'w') as out_file:
		fieldnames = ['lemmatized', 'id', 'conversion', 'tags', 'acm', 'action', 'agregator', 'amn', 'city', 'contacts', 'date', 'geo', 'poi', 'price', 'stars', 'time', 'wh', 'other']
		writer = csv.DictWriter(out_file, fieldnames=fieldnames, doublequote=False, escapechar='\\')
		writer.writeheader()

		for row in reader:
			#print(row)
			try:
				try:
					if int(row['id']) % 500 == 0:
						print(row['id'])
				except TypeError:
					pass
				tokens = row['lemmatized'].split()
				row_out = {}
				row_out['lemmatized'] = row['lemmatized']
				row_out['id'] = row['id']
				row_out['conversion'] = row['conversion']
				row_tags = []
				for entry in tags_dict:
					if entry in row['lemmatized']:
						row_tags.append(tags_dict[entry])
					if entry in row['keyword']:
						row_tags.append(tags_dict[entry])
				row_tags = list(set(row_tags))
				row_out['tags'] = ' '.join(row_tags)
				row_out['other'] = ''
				for i in row_tags:
					in_facets = False
					for facet in facets:
						if i.strip().startswith(facet) and not (i.strip() == facet):
							in_facets = True
							if facet in row_out:
								row_out[facet] += ' ' + re.sub(facet, '', i)
							else:
								row_out[facet] = re.sub(facet, '', i)
					if not in_facets:
						row_out['other'] += i + ' '
				writer.writerow(row_out)
			except AttributeError:
				pass
			#print(tokens)