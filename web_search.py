import os
import datetime as _dt
from urllib.parse import urlparse

def _has_openai(): return bool(os.getenv("OPENAI_API_KEY"))
def _has_tavily(): return bool(os.getenv("TAVILY_API_KEY"))


class CreditCardWebSearch:
    """
    Web search using tavily-python when available, with LLM (OpenAI) summarization.
    Falls back to LangChain integration if necessary. If neither is available,
    returns an empty string.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", max_results: int = 6):
        self.max_results = max_results
        self.has_tavily = _has_tavily()
        self.has_openai = _has_openai()
        self._tavily_client = None
        if self.has_tavily:
            try:
                from tavily import TavilyClient  # type: ignore
                self._tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            except Exception:
                self._tavily_client = None
        # optional LLM client
        self._llm = None
        if self.has_openai:
            try:
                from openai import OpenAI  # type: ignore
                self._llm = OpenAI()
                self._llm_model = model_name
            except Exception:
                self._llm = None

    def _summarize(self, query: str, bullets: list[dict]) -> str:
        # format bullets for LLM
        year = _dt.datetime.now().year
        lines = []
        for b in bullets[: self.max_results]:
            title = b.get("title") or b.get("content") or ""
            url = b.get("url") or b.get("source") or ""
            parsed = urlparse(url)
            host = parsed.netloc.replace("www.", "") if parsed.netloc else ""
            lines.append(f"- {title.strip()} ({host})")
        base = (
            "Summarize recent India-focused credit card info for the user query.\n"
            "- Use trusted sources (issuer sites, RBI, reputed media).\n"
            "- Return 4–6 concise bullets with bank/site names; include year if available.\n"
            "- If no relevant sources found, say so briefly.\n\n"
            f"Query: {query}\n\n"
            + "Sources:\n" + "\n".join(lines)
        )
        if self._llm is not None:
            try:
                out = self._llm.chat.completions.create(
                    model=getattr(self, "_llm_model", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": base}],
                    temperature=0.2,
                    max_tokens=500,
                )
                return out.choices[0].message.content or ""
            except Exception:
                pass
        # no LLM; return raw bullets
        return "\n".join(lines)

    def search_credit_card(self, query: str) -> str:
        if not self.has_tavily:
            return ""
        try:
            if self._tavily_client is not None:
                res = self._tavily_client.search(query=query, max_results=self.max_results, include_answer=False)
                results = res.get("results", []) if isinstance(res, dict) else []
                return self._summarize(query, results)
        except Exception:
            pass
        # Fallback via LangChain tool if available
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
            bullets = TavilySearchResults(max_results=self.max_results).run(query)
            # bullets may be a string already
            if isinstance(bullets, str):
                return bullets
            if isinstance(bullets, list):
                return self._summarize(query, bullets)
        except Exception as e:
            return f"_Web search unavailable: {e}_"
        return ""
