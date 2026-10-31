from types import SimpleNamespace

from vector.generate import OpenAIAnswerer, Source, build_prompt, parse_citations

SOURCES = [
    Source("d1", "B12 and homocysteine", "Vitamin B12 deficiency raises homocysteine."),
    Source("d2", "Folate trial", "Folic acid lowered homocysteine by 25%."),
]


class FakeClient:
    """Stands in for openai.OpenAI(): records the request, returns a canned reply."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_prompt_numbers_sources_from_one():
    prompt = build_prompt("Does B12 matter?", SOURCES)
    assert "[1] B12 and homocysteine" in prompt and "[2] Folate trial" in prompt
    assert prompt.endswith("Question: Does B12 matter?")


def test_citations_are_checked_against_the_sources():
    cited, invalid = parse_citations("Yes [2]. Also [1][2]. And [7].", SOURCES)
    assert [s.doc_id for s in cited] == ["d2", "d1"]  # order of first citation, no repeats
    assert invalid == [7]  # cites a source that doesn't exist


def test_answerer_sends_sources_and_parses_reply():
    client = FakeClient("B12 deficiency raises homocysteine [1], and folate lowers it [2]. See [3].")
    answer = OpenAIAnswerer(model="test-model", client=client).answer("Does B12 matter?", SOURCES)
    request = client.requests[0]
    assert request["model"] == "test-model" and request["temperature"] == 0
    assert "[2] Folate trial" in request["messages"][1]["content"]
    assert [s.doc_id for s in answer.cited] == ["d1", "d2"] and answer.invalid_citations == [3]


def test_no_sources_means_no_model_call():
    client = FakeClient("should not be used")
    answer = OpenAIAnswerer(client=client).answer("anything", [])
    assert client.requests == [] and answer.cited == []
