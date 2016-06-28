#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv

bow_dict = {}

with open('Bag_of_Words_model_test.csv') as bag_of_words:
	reader_bow = csv.DictReader(bag_of_words, doublequote=False, escapechar='\\')
	for row in reader_bow:
		try:
			bow_dict[int(float(row['id']))] = int(float(row['conversion']))
		except ValueError:
			pass

#print(bow_dict[99992])

with open('data.csv/data_lem_tags_test.csv') as in_file:
	reader = csv.DictReader(in_file, doublequote=False, escapechar='\\')
	with open('data.csv/data_lem_tags_test_bow.csv', 'w') as out_file:
		fieldnames = reader.fieldnames
		fieldnames.append('bag_of_words')
		writer = csv.DictWriter(out_file, fieldnames=fieldnames, doublequote=False, escapechar='\\')
		writer.writeheader()
		for row in reader:
			try:
				row['bag_of_words'] = bow_dict[int(row['id'])]
				writer.writerow(row)
			except KeyError:
				print row['id']
				#pass