
def search_vector_db(query, embed_model, collection, k=5):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results["documents"][0]
