pipeline = RAGIngestionPipeline(
    data_dir="./data",
    qdrant_path="./qdrant_data",
    collection_name="my_documents",
)

pipeline.run()
