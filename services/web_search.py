import re


IMAGE_ARTIFACTS = re.compile(r"\S+\.(?:jpg|jpeg|png|gif|webp)\)?", re.IGNORECASE)
MARKDOWN_HEADERS = re.compile(r"^#{1,6}\s*", re.MULTILINE)
NOISY_TOKENS = re.compile(r"\b(?:max_bytes\([^)]*\)|strip_icc\(\)|data:image/\S+)\b", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")



def _clean_web_text(text):
    cleaned = (text or "").replace("\x00", " ")
    cleaned = MARKDOWN_HEADERS.sub("", cleaned)
    cleaned = NOISY_TOKENS.sub(" ", cleaned)
    cleaned = IMAGE_ARTIFACTS.sub(" ", cleaned)
    cleaned = WHITESPACE.sub(" ", cleaned).strip()
    return cleaned[:1400]



def web_search(query, tavily):
    if not query.strip():
        return []

    results = tavily.search(query=query, max_results=5)
    items = []
    for result in results.get("results", []):
        content = _clean_web_text(result.get("content") or "")
        if not content:
            continue
        items.append(
            {
                "content": content,
                "metadata": {
                    "title": (result.get("title") or "Web result").strip(),
                    "url": result.get("url") or "",
                },
                "distance": None,
            }
        )

    return items
