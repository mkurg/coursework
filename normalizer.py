#!/usr/bin/python
# -*- coding: utf-8 -*-

import pymorphy2
import csv

morph = pymorphy2.MorphAnalyzer()

counter = 0

with open('data.csv/data.csv') as data_file:
	reader = csv.DictReader(data_file, doublequote=False, escapechar='\\')
	with open('data.csv/data_lem.csv', 'w') as out_file:
		fieldnames = reader.fieldnames
		fieldnames.append('lemmatized')
		fieldnames.append('id')
		fieldnames.append('conversion')
		writer = csv.DictWriter(out_file, fieldnames=fieldnames, doublequote=False, escapechar='\\')
		writer.writeheader()
		for row in reader:
			if counter % 1000 == 0:
				print(counter)
			counter += 1
			#print(row['keyword'])
			try:
				a = row['keyword'].split()
				b = []
				for word in a:
					b.append(morph.parse(word)[0].normal_form)
				b = set(b)
				row['lemmatized'] = " ".join(b)
				row['id'] = counter
				if (int(row['conversion1']) >= 1) or (int(row['conversion2']) >= 1) or (int(row['conversion3']) >= 1):
					row['conversion'] = 1
				else:
					row['conversion'] = 0
				try:
					writer.writerow(row)
				except ValueError:
					print(row)
			except AttributeError:
				print(row)
			except TypeError:
				print(row)
			except ValueError:
				print(row)
			
			#print(row)