#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:23:56 2022

@author: lopid
"""

import pandas as pd

doc2=pd.read_csv("abstracts5.csv",sep=",", on_bad_lines='warn', index_col=False)
doc2.columns=['Journal','Title','Authors','Author_Information','Abstract','DOI','Misc']

Abstract = []

for i in range (len(doc2)):
	if len(list(doc2['Abstract'][i:i+1])[0])>100:
		Abstract.append(list(doc2['Abstract'][i:i+1])[0])

df = pd.DataFrame(Abstract)

df.to_csv('Abstract5.csv')
		


#doc2 = pd.read_csv("abstracts3.csv")