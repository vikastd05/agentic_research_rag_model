import re

from pypdf import PdfReader


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_END_RE = re.compile(r"[.!?]\s")



def normalize_text(text):
    return WHITESPACE_RE.sub(" ", (text or "")).strip()



def split_text(text, chunk_size=500, overlap=80, min_chunk_size=120):
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = target_end

        if end < text_length:
            sentence_matches = list(SENTENCE_END_RE.finditer(text, start, target_end))
            if sentence_matches:
                end = sentence_matches[-1].end() - 1
            else:
                space_break = text.rfind(" ", start, target_end)
                if space_break > start:
                    end = space_break

        chunk = text[start:end].strip()
        if chunk and (not chunks or chunk != chunks[-1]):
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - overlap, start + 1)
        remaining = text_length - next_start
        if remaining < min_chunk_size:
            tail = text[next_start:].strip()
            if tail:
                if chunks:
                    merged_tail = f"{chunks[-1]} {tail}".strip()
                    chunks[-1] = merged_tail
                else:
                    chunks.append(tail)
            break

        start = next_start

    cleaned_chunks = []
    for chunk in chunks:
        normalized_chunk = normalize_text(chunk)
        if normalized_chunk and (not cleaned_chunks or normalized_chunk != cleaned_chunks[-1]):
            cleaned_chunks.append(normalized_chunk)

    return cleaned_chunks



def ingest_pdf(file, embed_model, collection):
    file.seek(0)
    reader = PdfReader(file)

    # Replace any existing chunks for this file so stale fragments are not retrieved.
    try:
        collection.delete(where={"file_name": file.name})
    except Exception:
        pass

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        chunks = split_text(text)

        for i, chunk in enumerate(chunks):
            embedding = embed_model.encode(chunk).tolist()
            chunk_id = f"{file.name}_{page_num}_{i}"
            collection.upsert(
                documents=[chunk],
                embeddings=[embedding],
                ids=[chunk_id],
                metadatas=[{"page": page_num, "file_name": file.name}],
            )

    file.seek(0)
