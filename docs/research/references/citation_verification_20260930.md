# Citation Verification Notes (2026-09-30)

This file records how the local bibliography was checked. It is not a substitute for the authors' final human citation review.

| Key | Source used | Verified fields | Remaining author check |
|---|---|---|---|
| `park2024self` | Repository README and linked OpenReview page | title, authors, ICLR 2024, URL | Confirm final venue citation style and whether the accepted-paper record should replace the repository BibTeX |
| `cai2024mambats` | Official MambaTS README and linked arXiv page | title, authors, arXiv id, year, official repository URL | Confirm the final published/archival version if one is cited instead of arXiv |
| `ahamed2024timemachine` | Crossref DOI extraction plus official TimeMachine README | title, authors, ECAI 2024, IOS Press, DOI, ISBN, ISSN, year | Confirm venue-required proceedings style and optional page range at submission time; OpenAlex's aggregate journal fields were not used |
| `karadag2026msmamba` | Crossref DOI extraction plus official ms-Mamba README | title, authors, journal, volume, article number, DOI, year | Confirm publisher version and repository license before redistributing code |
| `automl2024mamba4cast` | Official GitHub README | software title, organization, repository URL, access date | Add a peer-reviewed paper citation only if the authors verify one; no external numerical result is used locally |

## Scope

The repository registry distinguishes structural references from local experiments. The local implementation is not a bitwise reproduction of any external repository, and public repository benchmark values are not copied into the manuscript's result tables. The exact external commit and license text should be recorded by the human authors before submission.
