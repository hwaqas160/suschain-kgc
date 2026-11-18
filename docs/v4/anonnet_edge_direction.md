# AnonNet edge direction

The AnonNet release (`edges_anon.csv`: columns `source`, `target`) does not say
whether `source` supplies `target` or the reverse. The substitution task needs
to know which side is the supplier, so we checked the data (17 November 2026).

| Check | Result | Reading under "source supplies target" |
|---|---|---|
| Firms with the most outgoing links | power semiconductors for vehicle ECUs (483), remelt (418), metallurgical coal and limestone (380), coal and uranium (362) | upstream producers supply many firms: consistent |
| Upstream product keywords (steel, resin, chemical, metal, rubber, fertilizer, wire, bearings, fasteners ...) | 26.5% of firms with outgoing links, 19.6% of firms with none | suppliers are more often upstream: consistent |
| Links where only the source is upstream / only the target is | 18.6% / 15.9% | consistent, weakly |
| Links where only the target sells finished goods / only the source does | 2.5% / 0.7% | finished-goods firms receive links 3.5 times as often as they send them: consistent |

Every check points the same way, but none is conclusive, and some
high-in-degree firms (insurance, real estate services) are not obvious
customers. The paper must state the reading as an assumption supported by these
checks, and the dataset's authors should be asked to confirm it.

Reproduce: the keyword lists and counts are in the session script used to
produce this table; re-run against `nodes_anon.csv` and `edges_anon.csv`.
