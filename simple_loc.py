import random
from sentence_transformers import SentenceTransformer, util
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')

#Unfortunatly testing has disproven this as an effective method given the model
#Better results may be found in a madle trained specificly with our image set

def a_test(img_set):
    def search(query, k=3):
        # First, we encode the query (which can either be an image or a text string)
        query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        
        # Then, we use the util.semantic_search function, which computes the cosine-similarity
        # between the query embedding and all image embeddings.
        # It then returns the top_k highest ranked images, which we output
        hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
        
        print("Query:")
        for hit in hits:
            print(img_paths[hit['corpus_id']])
            
    i = 0
    img_paths = []
    while i<71:
        img_paths.append(f'data\\{img_set}\\{i}.jpg')
        i= i + 2
    
    img_emb = model.encode([Image.open(filepath) for filepath in img_paths], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
    test_img = f'{random.randint(0,71)}.jpg'
    search(Image.open('data\\0\\1.jpg'))
    print(f'Image Given: {test_img}')

for i in range(10):           
    a_test(i)