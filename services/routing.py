TEMPORAL_WEB_SIGNALS = [
    "latest",
    "today",
    "current",
    "news",
    "recent",
    "stock",
    "price",
    "weather",
    "who is",
    "what happened",
    "live",
    "breaking",
]

DOCUMENT_SIGNALS = [
    "paper",
    "pdf",
    "document",
    "research",
    "uploaded file",
    "this file",
    "this pdf",
    "in the file",
    "from the document",
    "from the pdf",
    "summarize the file",
    "as per the uploaded pdf",
]



def route_query(query, has_documents=False):
    text = (query or "").strip().lower()
    if not text:
        return "NONE"

    if any(signal in text for signal in DOCUMENT_SIGNALS):
        return "DOCUMENT" if has_documents else "WEB"

    if any(signal in text for signal in TEMPORAL_WEB_SIGNALS):
        return "WEB"

    if has_documents:
        return "AUTO"

    return "WEB"
