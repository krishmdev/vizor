from vizor.ingest.extract import extract

HTML = """<!doctype html><html><head><title>Aria | Crema Lab</title>
<meta name="description" content="Single boiler with PID.">
<link rel="canonical" href="https://crema-lab.example/aria">
<script type="application/ld+json">{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[
{"@type":"Question","name":"Is it quiet?",
 "acceptedAnswer":{"@type":"Answer","text":"Yes, at 52 dB."}}]}</script>
</head><body><nav><a href="/">Home</a></nav><main><h1>Aria</h1>
<p>The Aria is a single boiler espresso machine with PID temperature control and a 58 mm group.
It heats up in five minutes and holds temperature within a degree across a session of shots.</p>
<h3>Is it quiet?</h3><p>Yes, at 52 dB.</p>
<ul><li><a href="/aria-dual">Aria Dual</a></li><li><a href="https://other.example/x">Other</a></li></ul>
</main><footer>fictional</footer></body></html>"""


def test_extract_fields():
    d = extract(HTML, "https://crema-lab.example/aria?ref=x")
    assert d.url == "https://crema-lab.example/aria"
    assert d.domain == "crema-lab.example"
    assert d.title == "Aria | Crema Lab"
    assert d.meta_description == "Single boiler with PID."
    assert d.faq == (("Is it quiet?", "Yes, at 52 dB."),)
    assert d.jsonld[0]["@type"] == "FAQPage"
    assert d.links == (("Aria Dual", "https://crema-lab.example/aria-dual"),)
    assert "single boiler espresso machine" in d.body
    assert "52 dB" not in d.body
    assert "fictional" not in d.body


def test_faq_from_headings_without_jsonld():
    html = HTML.replace('"@type":"FAQPage"', '"@type":"WebPage"')
    d = extract(html, "https://crema-lab.example/aria")
    assert d.faq == (("Is it quiet?", "Yes, at 52 dB."),)


def test_demo_corpus_loads(docs, project):
    assert len(docs) == 22
    assert {d.domain for d in docs} == set(project.domains)
    targets = [d for d in docs if d.role == "target"]
    assert len(targets) == 5
    assert not any(d.faq or d.jsonld or d.links for d in targets)
