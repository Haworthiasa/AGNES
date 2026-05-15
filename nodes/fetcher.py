"""Fetch and extract arXiv papers."""

from pathlib import Path
from urllib.parse import urlparse

from config import OUTPUT_DIR
from state import PaperState


def extract_arxiv_id(url: str) -> str | None:
    """Extract an arXiv ID from an arXiv URL or raw arXiv ID."""
    value: str = url.strip()
    if not value:
        return None

    if value.startswith(("arxiv.org/", "www.arxiv.org/")):
        value = f"https://{value}"

    parsed = urlparse(value)
    if parsed.netloc:
        path_parts: list[str] = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"abs", "pdf"}:
            arxiv_id: str = "/".join(path_parts[1:])
            if arxiv_id.endswith(".pdf"):
                arxiv_id = arxiv_id[:-4]
            return arxiv_id or None
        return None

    if value.endswith(".pdf"):
        value = value[:-4]

    return value or None


def fetch_paper(state: PaperState) -> dict[str, str]:
    """Download an arXiv paper PDF and extract its text content."""
    arxiv_url: str = state["arxiv_url"]
    arxiv_id: str | None = extract_arxiv_id(arxiv_url)

    if arxiv_id is None:
        error_message: str = f"Error: could not extract arXiv ID from {arxiv_url!r}"
        print(error_message)
        return {
            "paper_content": error_message,
            "paper_title": error_message,
        }

    print(f"Fetching arXiv paper: {arxiv_id}")

    try:
        import arxiv
        import certifi
        import requests
        from pypdf import PdfReader

        output_dir: Path = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search), None)

        if paper is None:
            error_message = f"Error: no arXiv paper found for ID {arxiv_id!r}"
            print(error_message)
            return {
                "paper_content": error_message,
                "paper_title": error_message,
            }

        safe_arxiv_id: str = arxiv_id.replace("/", "_")
        pdf_path: Path = output_dir / f"{safe_arxiv_id}.pdf"

        print(f"Downloading PDF to {pdf_path}")
        response = requests.get(
            paper.pdf_url,
            timeout=60,
            verify=certifi.where(),
        )
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

        print(f"Extracting text from {pdf_path}")
        reader = PdfReader(str(pdf_path))
        page_text: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text: str = page.extract_text() or ""
            if text.strip():
                page_text.append(text)
            else:
                print(f"No text extracted from page {page_number}")

        paper_content: str = "\n\n".join(page_text).strip()
        if not paper_content:
            error_message = f"Error: no text could be extracted from {pdf_path}"
            print(error_message)
            return {
                "paper_content": error_message,
                "paper_title": paper.title,
            }

        print(f"Fetched paper title: {paper.title}")
        return {
            "paper_content": paper_content,
            "paper_title": paper.title,
        }
    except Exception as exc:
        error_message = f"Error fetching paper {arxiv_id!r}: {exc}"
        print(error_message)
        return {
            "paper_content": error_message,
            "paper_title": error_message,
        }
