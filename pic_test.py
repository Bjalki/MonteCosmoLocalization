from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os

#Testing indicates that the model can detect images from near angles but given the similarities in our images
#It cannot do so with enough consistency without training on a cosmo data set 

model = SentenceTransformer('clip-ViT-B-32')

#Encode an image:
img_emb = model.encode(Image.open('data\\0\\0.jpg'))
img_emb2 = model.encode(Image.open('data\\0\\1.jpg'))

img_emb3 = model.encode(Image.open('data\\0\\23.jpg'))


img_emb4 = model.encode(Image.open('data\\0\\2.jpg'))
#Compute cosine similarities 
print('Similar Images')
cos_scores = util.cos_sim(img_emb, img_emb2)
print(cos_scores)

print()
print("Different Images")
cos_scores = util.cos_sim(img_emb, img_emb3)
print(cos_scores)

print()
print('Should be slightly lower than 1')
cos_scores = util.cos_sim(img_emb, img_emb4)
print(cos_scores)


