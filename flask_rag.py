from flask import Flask, request, jsonify
from src.chroma import get_vectorstore
from src.llm import get_qa_chain


app = Flask(__name__)
vectorstore = get_vectorstore()
qa_chain = get_qa_chain(vectorstore, rerank='cross')


@app.route('/genetate_text', methods=['POST'])
def genetate_text():
    data = request.json
    promt = data['promt']

    result = qa_chain.invoke(promt)
    answer = result.get("result", "Ответ не найден.")
    sources_raw = result.get("source_documents", [])

    seen = set()
    source_count = 0
    sources = ""
    for source in sources_raw:
        book_title = source.metadata.get('book_title', 'Unknown')
        chapter_title = source.metadata.get('chapter_title', 'Unknown')

        unique_key = (book_title, chapter_title)

        if unique_key not in seen:
            source_count += 1
            seen.add(unique_key)
            sources += f"{source_count}. {book_title}, {chapter_title}\n"

    return jsonify({"text": answer, "sources": sources})


if __name__=="__main__":
    app.run(port=8000)
