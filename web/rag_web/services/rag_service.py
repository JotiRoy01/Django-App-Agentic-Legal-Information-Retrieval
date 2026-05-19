from agentic.pipeline.rag_orchestrator import RAGPipelineOrchestrator


class RAGService:

    def __init__(self):
        self.orchestrator = RAGPipelineOrchestrator()

    def ask(self, question: str):

        response = self.orchestrator.ask(question)

        return response