from vizor.metrics.mentions import brand_patterns, brand_regex, default_names, mention_counts
from vizor.types import CitedSentence


def test_brand_names_match_whole_words_case_insensitively():
    r = brand_regex(["Cog & Chain", "Brewline"])
    assert r.findall("The Cog and Chain Urbano beats the BREWLINE Duo.") == [
        "Cog and Chain",
        "BREWLINE",
    ]
    assert r.search("cog & chain") and r.search("Brewline's pump")
    assert not r.search("Brewlines are great") and not r.search("cogwheel")


def test_default_name_reads_hyphens_as_optional_spaces():
    assert default_names("crema-lab.example") == ["crema lab"]
    pat = brand_patterns(["crema-lab.example"])["crema-lab.example"]
    for text in ("Crema Lab Aria", "crema-lab.example/aria", "the CremaLab site"):
        assert pat.search(text), text


def test_project_brands_override_the_default():
    pats = brand_patterns(["shotreport.example"], {"shotreport.example": ["Shot Report"]})
    assert pats["shotreport.example"].search("measured by Shot Report")


def test_mentions_ignore_citation_markers():
    pat = brand_regex(["Source"])
    s = [CitedSentence(0, "It is quiet [Source 2].", 3, (2,))]
    assert mention_counts(s, pat) == (0, 0)
    pat = brand_regex(["Brewline"])
    s = [
        CitedSentence(0, "The Brewline Duo and the Brewline Solo [1].", 7, (1,)),
        CitedSentence(1, "Neither is cheap.", 3, ()),
    ]
    assert mention_counts(s, pat) == (1, 2)


def test_demo_project_lists_brands(project):
    pats = project.brand_patterns()
    assert set(pats) == set(project.domains)
    assert pats["shotreport.example"].search("Shot Report measured it")
