# Bike commuting bench corpus

Everything in this folder is synthetic. The five sites are fictional brands on RFC 2606 `.example`
domains, and all product names (Larkspur Metro 7, Cog & Chain Pocket 20, and others), prices,
specs and test numbers are invented.

The demo corpus in `data/demo` has only five target pages, which is too few independent units
for page-level statistics. This corpus was written for Vizor to give the sandbox 24 target pages
in a different vertical: bicycle commuting, bikes and bike maintenance.

| Domain | Role | Pages | Character |
|---|---|---|---|
| larkspur.example | target | 24 | Bike maker with gaps on purpose: title is just the page name, weak or missing meta description, no FAQ, no JSON-LD, no internal links, key numbers in the last paragraphs |
| cogandchain.example | competitor | 12 | Bike maker with descriptive titles and meta descriptions, Product or Article plus FAQPage JSON-LD, an on-page FAQ and related links |
| spokewise.example | competitor | 12 | Review and test site heavy on measured numbers, often comparing named products |
| pedalcheap.example | competitor | 12 | Thin affiliate pages with clickbait titles and few facts |
| saddlesore.example | competitor | 12 | First-person hobbyist blog with practical tips |

There are 24 topics, 12 products and 12 guides. Each topic has exactly one Larkspur page and two
relevant pages from two different competitor sites.

`queries.jsonl` contains 72 queries, 3 per topic and 18 for each intent (informational,
comparison, transactional, troubleshooting). They were written by hand. None of them names
Larkspur, and only two name a product at all. Each query has a `topic` field holding the slug of
the Larkspur page for its topic (for example `metro-7` or `guides/flat-tire`), so results can be
grouped by target page. The loader ignores `topic`; it uses the `brands` key in
`project.yaml` to score whole-word brand mentions.
