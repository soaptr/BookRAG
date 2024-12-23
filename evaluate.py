import json
import time
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted
from src.chroma import get_vectorstore
from src.llm import get_qa_chain


def prepare_rag_answers(qa_chain, examples):
    """
    Подготавливает ответы от RAG-системы для списка вопросов.

    Args:
        qa_chain (RetrievalQA): Объект RAG-системы, используемый для генерации ответов.
        examples (list): Список примеров в формате [(вопрос, правильный ответ), ...].

    Returns:
        list: Список ответов от RAG-системы для каждого вопроса.
    """
    answers = []

    for qa_pair in examples:
        while True:
            try:
                answer = qa_chain.invoke({"query": qa_pair['question']})
            except ResourceExhausted as e:
                print(f"An exception of type {type(e).__name__} occurred. Wait for 5 seconds.")
                time.sleep(5)
                continue
            break
        answers.append(answer)

    return answers


def evaluate_accuracy(examples, student_answers, llm):
    """
    Оценка точности (accuracy) ответов.
    """
    grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")
    answer_grader = grade_prompt_answer_accuracy | llm

    scores = []
    for example, student_answer in zip(examples, student_answers):
        while True:
            try:
                score = answer_grader.invoke({
                    "question": example['question'],
                    "correct_answer": example['answer'],
                    "student_answer": student_answer['result']
                })[0]['args']['Score']
            except ResourceExhausted as e:
                print(f"An exception of type {type(e).__name__} occurred. Wait for 5 seconds.")
                time.sleep(5)
                continue
            break
        scores.append(score)

    return scores


def evaluate_helpfulness(examples, student_answers, llm):
    """
    Оценка полезности (helpfulness) ответов.
    """
    grade_prompt_answer_helpfulness = hub.pull("langchain-ai/rag-answer-helpfulness")
    answer_grader = grade_prompt_answer_helpfulness | llm

    scores = []
    for example, student_answer in zip(examples, student_answers):
        while True:
            try:
                score = answer_grader.invoke({
                    "question": example['question'],
                    "student_answer": student_answer['result']
                })[0]['args']['Score']
            except ResourceExhausted as e:
                print(f"An exception of type {type(e).__name__} occurred. Wait for 5 seconds.")
                time.sleep(5)
                continue
            break
        scores.append(score)

    return scores


def evaluate_hallucinations(examples, student_answers, llm):
    """
    Оценка уровня галлюцинаций (hallucination) в ответах.
    """
    grade_prompt_hallucinations = hub.pull("langchain-ai/rag-answer-hallucination")
    answer_grader = grade_prompt_hallucinations | llm

    scores = []
    for example, student_answer in zip(examples, student_answers):
        while True:
            try:
                score = answer_grader.invoke({
                    "documents": student_answer['source_documents'],
                    "student_answer": student_answer['result']
                })[0]['args']['Score']
            except ResourceExhausted as e:
                print(f"An exception of type {type(e).__name__} occurred. Wait for 5 seconds.")
                time.sleep(5)
                continue
            break
        scores.append(score)

    return scores


def evaluate_document_relevance(examples, student_answers, llm):
    """
    Оценка релевантности документов (document relevance).
    """
    grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")
    answer_grader = grade_prompt_doc_relevance | llm

    scores = []
    for example, student_answer in zip(examples, student_answers):
        while True:
            try:
                score = answer_grader.invoke({
                    "question": example['question'],
                    "documents": student_answer['source_documents']
                })[0]['args']['Score']
            except ResourceExhausted as e:
                print(f"An exception of type {type(e).__name__} occurred. Wait for 5 seconds.")
                time.sleep(5)
                continue
            break
        scores.append(score)

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG-based answers.")
    parser.add_argument(
        "--rerank",
        type=str,
        choices=["flash", "cross", None],
        default=None,
        help="Модель для повторного ранжирования ('flash', 'cross' или None)"
    )
    args = parser.parse_args()

    vectorstore = get_vectorstore()
    qa_chain = get_qa_chain(vectorstore, rerank=args.rerank)
    llm_judge = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    with open("data/questions.json", "r", encoding="utf-8") as file:
        questions = json.load(file)

    answers = prepare_rag_answers(qa_chain, questions)

    # Оценка
    accuracy_scores = evaluate_accuracy(questions, answers, llm_judge)
    helpfulness_scores = evaluate_helpfulness(questions, answers, llm_judge)
    hallucinations_scores = evaluate_hallucinations(questions, answers, llm_judge)
    document_relevance_scores = evaluate_document_relevance(questions, answers, llm_judge)

    # Результаты
    print(f"Accuracy Scores: {accuracy_scores}")
    print(f"Helpfulness Scores: {helpfulness_scores}")
    print(f"Hallucinations Scores: {hallucinations_scores}")
    print(f"Document Relevance Scores: {document_relevance_scores}")

    accuracy = sum(accuracy_scores) / len(accuracy_scores)
    helpfulness = sum(helpfulness_scores) / len(helpfulness_scores)
    hallucinations = sum(hallucinations_scores) / len(hallucinations_scores)
    document_relevance = sum(document_relevance_scores) / len(document_relevance_scores)

    print(f"Accuracy: {accuracy}")
    print(f"Helpfulness: {helpfulness}")
    print(f"Hallucinations: {hallucinations}")
    print(f"Document Relevance: {document_relevance}")
