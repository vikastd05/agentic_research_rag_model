def _build_context_block(context):
    blocks = []
    for index, item in enumerate(context, start=1):
        metadata = item.get("metadata", {})
        label = metadata.get("file_name") or metadata.get("title") or f"Source {index}"
        blocks.append(f"[{index}] {label}\n{item.get('content', '')}")
    return "\n\n".join(blocks)



def _fallback_answer(context, question):
    if not context:
        return "I could not find supporting context for that question. Upload a PDF or check your API keys and try again."

    snippets = []
    for item in context[:2]:
        content = item.get("content", "").strip().replace("\n", " ")
        snippets.append(content[:280])

    joined = " ".join(snippets).strip()
    return f"Based on the available context, the most relevant information for '{question}' is: {joined}"



def generate_answer(context, question, llm, model_name):
    if not question.strip():
        return "Enter a question to begin."

    if not context:
        return "No relevant context was found for this question. Upload a document or try a broader query."

    prompt = f"""
You are a research assistant.
Answer the user's question using only the provided context.
If the context is incomplete, say so clearly.
Keep the answer concise and accurate.

Context:
{_build_context_block(context)}

Question:
{question}
"""

    try:
        response = llm.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You answer from provided context and do not invent facts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _fallback_answer(context, question)
