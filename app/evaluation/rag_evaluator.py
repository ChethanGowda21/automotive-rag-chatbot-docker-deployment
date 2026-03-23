class RAGEvaluator:

    def evaluate(self, question, answer, context):

        return {
            "context_used": len(context),
            "answer_length": len(answer)
        }