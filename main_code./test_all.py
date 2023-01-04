#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:23:56 2022

@author: lopid
"""

import pandas as pd
import numpy as np
import collections

import nltk
import numpy as np
from nltk.corpus import stopwords


class A_Matrix:
	def __init__(self, Themes,b_inf,b_sup,N,Words_to_remove):
		self.Themes= Themes
		self.b_inf= b_inf
		self.b_sup= b_sup
		self.N=N
		self.Words_to_remove=Words_to_remove

	def Get_Words_and_Abstracts(self):
		
		Abstract = []
		Words=[]
		List_abstracts=self.Themes
		for j in List_abstracts:
			doc= pd.read_csv("{}".format(j),sep=",", on_bad_lines='warn', index_col=False)
			doc.columns=['Index','Abstract']
			
			for i in range (self.N):

				character = list(doc['Abstract'][i:i+1])[0]
				character= character.replace('.', '')
				character= character.replace(':', '')
				character= character.replace(';', '')
				character= character.replace(',', '')
				character= character.replace("'", '')
				character= character.replace('"', '')
				character= character.replace('(', '')
				character= character.replace(')', '')
				Abstract.append(character)
				
				character= character.split()
				Words= Words + character
					
		Words = np.unique(np.array(Words))
		return(Words,Abstract)
	
	def Words_without_Stop_words(self):
		Stop_words = self.Words_to_remove
		Words = self.Get_Words_and_Abstracts()[0]
		Stop_words_del= []
		for k in range (len(Words)):
			if bool(np.isin(Words[k],Stop_words)):
				Stop_words_del.append(k)
		
		Words= np.delete(Words, Stop_words_del)
		return(Words)
	
	def Occurences(self):
		Words=self.Words_without_Stop_words()
		Old_W,Abstract= self.Get_Words_and_Abstracts()
		n = len(Words)
		m = len(Abstract)
		Word_by_doc = np.zeros((n,m)) #mots sur les lignes	
		
		occ=[]
		abstr=[]
				
		for i in range(len(Abstract)):			

			abstr= Abstract[i].split()
			count = collections.Counter(abstr)
			occ = [count[w] for w in Words]
			Word_by_doc [:, i]= occ
			
		return(Word_by_doc)


	def WBD_filtered(self):
		Word_by_doc = self.Occurences()
		Old_W,Abstract= self.Get_Words_and_Abstracts()
		n = len(Word_by_doc)
		m = len(Abstract)
		
		Mat_isin = np.ones((n,m))*(Word_by_doc!=0)	
		Percentage = np.sum(Mat_isin,axis=1)/m
		
		
		Borne = Percentage*(self.b_inf>=Percentage)*(Percentage>=self.b_sup)
		
		Out_of_percent = np.nonzero(Borne)[0]
		
		WBD_in_occ = np.delete(Word_by_doc, Out_of_percent,axis=0)
		
		Tot = np.sum(WBD_in_occ)
		WBD_in_freq = WBD_in_occ*(1/Tot)
		return(WBD_in_freq)
	
	
import math
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
from sklearn import cluster, datasets, manifold, decomposition, metrics
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm


class BiCoCluster :
	"""
	 Building cocluster requires attributes:
		 Adjacency matrix A
		 Number of cluster k
		 Colors we choose (if plots to see the clusters)
		 
	 """
	def __init__(self, A,k, colors):
		self.A = A
		self.k = k
		self.colors = colors
		
	
	def D1D2(self):
		# D1 D2 diagonal matrices
		# D1 : diagonal elements are sums over A's columns
		# D2 : diagonal elements are sums over A's lines
		D1 = np.diag(np.sum(self.A, axis=1))
		D2 = np.diag(np.sum(self.A, axis = 0))
		
		return(D1,D2)
	
	
	def Bipartitions(self):
		
		m,n= np.shape(self.A)
		# k clusters, l singular values
		l = int(math.log(self.k,2))+1
		
		
		D1,D2 = self.D1D2()
		B =sqrtm(np.linalg.inv(D1))
		C = sqrtm(np.linalg.inv(D2))
		
		# Find singular values of normalized adjacency matrix An
		An = np.dot(np.dot(B,self.A),C)
		U,sigma,V = np.linalg.svd(An)
		
		#linalg.svd already returns in descending order of singular values

		
		# Take  vectors number2
		Ul= np.dot( B , U[:,3] )
		Vl= np.dot( C, V[:,3] )
		X = np.append(Ul, Vl,axis=0)
		Y = np.array([0]*len(X))
		"""
		Z = np.zeros((len(X),2))
		Z[:,0]=X
		Z[:,1]=Y
		"""
		z2 = np.append(Ul,Vl).reshape(m+n,1)
		algorithm = cluster.KMeans(n_clusters=2)
		algorithm.fit(z2)
		
		
		if hasattr(algorithm,"labels__"):
			y_pred = algorithm.labels__astype(int)
			
		else:
			y_pred = algorithm.predict(z2)
			
		
		return(z2,y_pred)

	
	def WatchClustersBipart(self):
		#X = self.A[:,0]
		#Y = self.A[:,1]
		m,n = np.shape(self.A)
		col = self.colors
		Z,y_pred= self.Bipartitions()
		y_exp = np.array([0]*int(n/2)+[1]*int(n/2))
		
		
		us = np.array([1/2]*+int(n/2))
		algo = np.array([0]*int(n/2))
		
		x1 = np.arange(0,int(n/2))
		x2 = np.arange(int(n/2),n)
		
		plt.figure(figsize=(15,3))
		
		Predict1 = plt.scatter(x1, algo , marker="x", s=100, color=col[y_pred[m:m+int(n/2)]])
		Predict2 = plt.scatter(x2, algo ,  marker="x", s=100, color=col[y_pred[m+int(n/2):m+n]])
		Expect1 = plt.scatter(x1, us,s=100,color=col[y_exp[0:int(n/2)]])
		Expect2 = plt.scatter(x2, us,s=100,color=col[y_exp[int(n/2):n]])
		
		plt.legend(( Expect1,Expect2),
           ('Expected labels (label 1)','Expected labels (label 2)'),
           scatterpoints=1,
           loc='center',
           ncol=1,
           fontsize=14)
		plt.xlabel('Index of document')
		
		plt.show()
		
	def WatchWordsBipart(self):
		#X = self.A[:,0]
		#Y = self.A[:,1]
		m,n = np.shape(self.A)
		col = self.colors
		Z,y_pred= self.Bipartitions()
		algo = np.array([0]*m)
		
		x = np.arange(0,m)
		plt.figure(figsize=(15,1))
		plt.scatter(x, algo ,s=np.array(y_pred[0:m])*30+70, color=col[y_pred[0:m]], alpha=0.1)
		#plt.scatter(x, algo ,s=np.array(y_pred[0:m])*70+70, color=col[y_pred[0:m]], alpha=0.1)
		plt.xlabel('Index of word')
		plt.show()

class MultiCoCluster :
	"""
	 Building cocluster requires attributes:
		 Adjacency matrix A
		 Number of cluster k
		 Colors we choose (if plots to see the clusters)
		 
	 """
	def __init__(self, A, k, colors):
		self.A = A
		self.k = k
		self.colors = colors
		
	
	def D1D2(self):
		# D1 D2 diagonal matrices
		# D1 : diagonal elements are sums over A's columns
		# D2 : diagonal elements are sums over A's lines
		D1 = np.diag(np.sum(self.A, axis=1))
		D2 = np.diag(np.sum(self.A, axis = 0))
		
		return(D1,D2)
	
	def multipartitions(self):
		
		
		# k clusters, l singular values
		l = int(math.log(self.k,2))+1
		
		
		D1,D2 = self.D1D2()
		B =sqrtm(np.linalg.inv(D1))
		C = sqrtm(np.linalg.inv(D2))
		
		# Find singular values of normalized adjacency matrix An
		An = np.dot(np.dot(B,self.A),C)
		U,sigma,V = np.linalg.svd(An)
		
		#linalg.svd already returns in descending order of singular values

		
		# Take  vectors from 2 to l+1
		Ul= np.dot( B , U[:,2:l+2] )
		Vl= np.dot( C, V[:,2:l+2] )
		Z = np.append(Ul, Vl,axis=0)
		
		
		algorithm = cluster.KMeans(n_clusters=self.k)
		algorithm.fit(Z)
		
		
		if hasattr(algorithm,"labels__"):
			y_pred = algorithm.labels__astype(int)
			
		else:
			y_pred = algorithm.predict(Z)
			
			
		return(Z,y_pred)
	
	def WatchClustersMultipart(self):
		#X = self.A[:,0]
		#Y = self.A[:,1]
		col = self.colors
		Z,y_pred= self.multipartitions()
		m,n = np.shape(self.A)
		plt.figure(figsize=(15,15))
		plt.scatter(Z[:,0], Z[:,1] ,color=col[y_pred])
		plt.xlabel('First column of Z (size (m+n)*l)')
		plt.ylabel('Second column of Z (size (m+n)*l)')

		plt.show()
		
		
	

if __name__ == "__main__":
	
	"""
	Parameters for matrix A
	"""
	
	
	No_abs_per_theme= 25
	inf=0.01
	sup=0.17
	nltk.download('stopwords')
	Stop_words=stopwords.words('english')
	
	
	
	
	
	"""
	Bipartition
	"""
	
	List_abstracts1= ['blood_Abstracts.csv','laws_Abstracts.csv']
	MAT_bi = A_Matrix(List_abstracts1, inf, sup, No_abs_per_theme, Stop_words)
	
	W_b,Astract = MAT_bi.Get_Words_and_Abstracts()
	W_without_SW_b = MAT_bi.Words_without_Stop_words()
	Not_filtered_b = MAT_bi.Occurences()
	A_b = MAT_bi.WBD_filtered()
	
	colors_b = np.array(["#0343DF","#E50000"])
	
	CCB = BiCoCluster(A_b,2,colors_b)
	D1_b,D2_b = CCB.D1D2()
	Z_b,y_pred_b = CCB.Bipartitions()
	
	CCB.WatchClustersBipart()
	CCB.WatchWordsBipart()

	
	
	"""
	
	List_abstracts2= ['blood_Abstracts.csv','laws_Abstracts.csv','eco_Abstracts.csv']
	MAT_mult = A_Matrix(List_abstracts2, inf, sup, No_abs_per_theme, Stop_words)

	W_m,A_m = MAT_mult.Get_Words_and_Abstracts()
	W_without_SW_m = MAT_mult.Words_without_Stop_words()
	Not_filtered_m = MAT_mult.Occurences()

	A_m = MAT_mult.WBD_filtered()
	
	
	k = len(List_abstracts2)#Enter your number of cluster
	l = int(math.log(k,2))+1
	colors_m = np.array(["#0343DF","#E50000","#40E0D0"])#Enter your chosen colors -must be of k different colors



	CCM = MultiCoCluster(A_m,k,colors_m)
	D1_m,D2_m = CCM.D1D2()
	Z_m,y_pred_m = CCM.multipartitions()
	CCM.WatchClustersMultipart()

"""
	pass
