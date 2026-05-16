from agentic.data_loader.data_loader import load
from agentic.chunkings.production_chunker import ProductionChunkingPipeline


law = load("laws_de.csv")

pro_chunker = ProductionChunkingPipeline()
df = pro_chunker.chunk_laws_dataset(law)
print(df.head())