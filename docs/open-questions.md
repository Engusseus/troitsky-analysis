# Open questions for the organizers

Where the 2027 rulebook is ambiguous, bridgesim encodes the **stricter** reading and flags
the rule as *ambiguous* in the app. Ask the organizers
(competitions.troitsky@ecaconcordia.ca) and update `rules/troitsky_2027.yaml` with the
answer.

| # | Rule | Question | What bridgesim does now |
|---|---|---|---|
| 1 | §8.2.1.1 Span length | The text says the span is measured "from the middle of the piers", but the glossary defines the clear span as "the shortest longitudinal distance between supporting parts touching the table", and Figure 2 shows a 1000 mm *minimum clear span* (inner faces) and a 1200 mm *maximum span* (pier centres). Which distance is compared with 1000–1200 mm? | Checks both the inner-face clear span and the centre-to-centre span against the bands and keeps the worse result. |
| 2 | §8.2.1.3 Total length | Does the −10 pts for not fitting the press (> 1420 mm) stack with the −5 pts for exceeding 1350 mm? | Applies both (−15 total). |
| 3 | §8.2.1.3 / App. B | Appendix B shows the crusher with dimensions 137.0 cm and 142.0 cm. Is the clear opening between the press uprights 1370 mm or 1420 mm? | Uses 1420 mm from the rulebook text. |
| 4 | §8.1 A-frames | "Any bridge whose base/leg/pier is at an angle of more than 90°" is an A-frame. Measured from what? Is any inclined pier disqualified, or only outward-splayed ones? | Flags any pier more than 1° from vertical as a disqualification risk. |
| 5 | §8.2.2.2 / §8.5 | Must the 150 × 200 mm cart pass along the deck centreline, or anywhere across the deck? | Checks a 150 mm wide path centred on the deck centreline. |
| 6 | §8.9 Clear opening | Is "the centre of the bridge" the midpoint between the piers, or the middle of the deck? Must the opening be clear all the way up, or only above the deck? | Uses the midpoint between support centrelines; checks every member above the deck surface. |
| 7 | §12.5 Loading plate | The text says the plate is 90 mm × 200 mm; the Appendix B photo shows a 10 cm × 20 cm plate. Which is right, and is the plate placed on the deck by hand before the ram descends? | Uses 90 × 200 mm, placed on the deck. |
| 8 | §8.6 Clear span test | How wide (in Z) is the 1000 × 150 mm box, and must it pass between the piers only or under the whole bridge? | Checks a 1000 mm free length below 150 mm, between the inner faces of the supports. |
