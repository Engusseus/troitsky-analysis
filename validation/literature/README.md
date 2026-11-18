# Published coupon tests of craft sticks

Coupon tests of craft sticks published by others, put in the formats bridgesim reads, so
the model's coupons can be compared with them (`validation/literature_coupons.py`). They
are not a team's own sticks or glue: use them to see how far the shipped placeholder
values are from measured sticks, not to replace a team's coupon tests.

| File | What | Read by |
|---|---|---|
| `ens_paris_saclay/stick_tension.csv` | 76 single sticks pulled to failure (1.1) | `bridgesim calibrate` (f_t); `lab` does not simulate tension |
| `ens_paris_saclay/stick_bending.csv` | 3 single sticks, flat, 3-point bending on 100 mm (1.2) | `lab --data`, `calibrate` |
| `ens_paris_saclay/glue_laps.csv` | the one lap whose glue line failed (1.3) | `lab --data`, `calibrate` |
| `ens_paris_saclay/other/laminated_bending.csv` | edu_14235 Table 1 whole (1-3 sticks, glued or not, flat or on edge) and the four results the text gives outside it (1.2) | `literature_coupons.py` only |
| `ens_paris_saclay/other/wood_failed_laps.csv` | the two laps that broke in the wood: lower bounds for the glue (1.3) | nothing |
| `ens_paris_saclay/other/buckling.csv` | single-stick and I-beam buckling (1.3) | `literature_coupons.py` only |
| `ens_paris_saclay/other/team_summaries.yaml` | means and standard deviations of the six 2018-19 teams and of Move Your Bridge 2.0 (1.3, 1.4) | nothing |
| `cedarville_2022/summary.yaml` | means and ranges of 10 tension and 24 lap tests (2) | nothing |

bridgesim reads only the `*.csv` directly in the folder it is given, recognised by a word
of their name and their columns (`study.coupon_files`); `other/` is a subfolder, and its
CSVs have none of the templates' column sets, so `calibrate` and `lab` never take them.
`tests/test_literature_data.py` checks both.

Every number is a source's, with its page, table or figure in the file's `notes` or
`source` column. Nothing was interpolated or filled in. Values read off a figure say so.
Where a source is ambiguous or contradicts itself, both readings are written down (section
3); the one file that had to pick one (`glue_laps.csv`) says which and why.

**Licence.** The licence of the Eduscol resources was not verified (the pages saved for
this work state none). Only the measured numbers are reproduced here, with attribution
and short quotations of the sources' wording; no figure or file of the sources is copied.
The ASEE paper is copyright ASEE 2022; the same applies.

## 1. ENS Paris-Saclay (Eduscol, Culture Sciences de l'Ingénieur)

All in French. File names below are those the work was done from; the URLs are the public
ones. Sticks and glue of section 1.1 and 1.2 are not described beyond "bâtonnets en bois"
of "environ 11,5 cm": **no stick of these campaigns was measured, as far as the sources
say**. The only stick dimensions are the 2018-19 teams' mean of 50 sticks (1.3: 114 x 9.6
x 2.02 mm, 630 kg/m^3), used here for every ENS coupon.

### 1.1 Tension of 76 sticks (edu_14237)

X. Jourdain, H. Horsin Molinaro, "Caractérisation de la variabilité de la résistance en
traction de bâtonnets en bois", ENS Paris-Saclay / Eduscol 14237, 16 Jan 2023.
https://sti.eduscol.education.fr/sites/eduscol.education.fr.sti/files/ressources/pedagogiques/14237/14237-caracterisation-de-la-variabilite-de-la-resistance-en-traction-de-batonnets-en-bois-ensps_0.pdf
and its annex
https://sti.eduscol.education.fr/sites/eduscol.education.fr.sti/files/ressources/pedagogiques/14237/14237-annexes-caracterisation-de-la-variabilite-de-la-resistance-en-traction-de-batonnets-de-bois.zip
(`Annexe_Resultats_Exp_Traction_Fmax.csv`: 76 failure loads in N, first column; the two
Python scripts beside it were not run).

76 sticks pulled to failure ("traction pure jusqu'à rupture"); mean 1389 N, standard
deviation 339.7 N (337.5 as a population), 409-2204 N (p. 2; recomputed from the annex:
1388.7, 339.7, 409, 2204). The write-up fits a lognormal law: 5 % quantile 829.9 N (p. 3),
907.3 N fitted to the histogram (p. 5). Fig. 1 shows a universal testing machine; its
model, the grips, the rate and the gauge length are not stated. **Caveat**: the sticks'
section is not given; `stick_tension.csv` carries the 2018-19 mean (9.6 x 2.02 mm), so
71.6 MPa is the mean stress on that section, not on these sticks. Which batch they came
from (the dyed 2018-19 sticks, the undyed ones of Move Your Bridge 2.0, or another) is not
stated (the annex's files carry dates of January 2022, zip metadata only).

### 1.2 Three-point bending of 1, 2 and 3 sticks (edu_14235, edu_14236)

X. Jourdain, H. Horsin Molinaro, "Expérimenter la RdM avec le banc 3R Bed100 :
Caractérisation d'assemblages sollicités en flexion (bâtonnets bois)", ENS Paris-Saclay /
Eduscol 14235, 3 Jan 2023; and "... : Étude théorique d'assemblages sollicités en flexion
(bâtonnets bois)", Eduscol 14236, 9 Jan 2023 (beam theory only, no new data). The PDFs'
file names were not recorded; the resources' own pages, as the PDFs cite them:
https://eduscol.education.fr/sti/si-ens-paris-saclay/ressources_pedagogiques/experimenter-la-rdm-banc-3rbed100-caracterisation-dassemblages-sollicites-en-flexion
and
https://eduscol.education.fr/sti/si-ens-paris-saclay/ressources_pedagogiques/experimenter-la-rdm-avec-le-banc-3r-bed100-etude-theorique-dassemblages-sollicites-en-flexion
(files under https://sti.eduscol.education.fr/sites/eduscol.education.fr.sti/files/ressources/pedagogiques/14235/
and .../14236/).

3R Bed100 bench, supports 100 mm apart, displacement control at 0.1 mm/s (the settings
dialog, p. 3: stop force 1000 N, maximum travel 30 mm; the test file there is dated
2021_03_26), peak load recorded. One, two or three sticks lying flat ("posé sur axe
faible") or on edge ("axe fort"), stacked unglued or glued face to face; 3 tests per
configuration (4 for three glued flat). Glue not named; cure 21 days for two sticks, 7 days
for three (one pair at 50 minutes, left out). Table 1 (p. 13) lists 31 peak loads; the
text adds four (three glued pairs left out of the table, section 5.1, and a fourth test of
three unglued sticks on edge). All in `other/laminated_bending.csv`; the three single
sticks flat also in `stick_bending.csv`.

Caveats: the supports' and the nose's diameter is not stated (about 20 mm steel rollers in
the photos). One single stick reached 2.2 cm of deflection at failure (p. 3): far outside
small deflections, so the modulus of rupture from F L / 4 over b t^2 / 6 (111-192 MPa) is
not a material strength. On edge, single and unglued sticks failed by lateral-torsional
buckling ("déversement"), not by strength. Glued stacks dented under the nose
("matriçage"). The curves (Fig. 4-26) have unlabelled axes and the text calls the
abscissa time. Fig. 4's slope (about 3 N per unit, read from the figure) and its 2.2 cm
deflection fit millimetres (E about 10 GPa on the nominal section; seconds at 0.1 mm/s
would give 100 GPa); Fig. 15's (about 4 N per unit) fits seconds (about 16 GPa;
millimetres would give 1.6 GPa). The figures do not share one abscissa: no stiffness is
taken from them. The source itself says three tests are too few for representative means
(p. 13).

### 1.3 Move Your Bridge 2018-19: six team reports (edu_11624)

H. Horsin Molinaro, X. Jourdain, M. Poncelet, "Move Your Bridge : défis étudiants", ENS
Paris-Saclay / Eduscol 11624, 2 Dec 2019, with the six teams' project reports (June 2019)
as annexes:
https://sti.eduscol.education.fr/sites/eduscol.education.fr.sti/files/ressources/pedagogiques/11624/11624-move-your-bridge-defis-etudiants-ensps.pdf
(`myb_main.pdf` here) and
`11624-annexe-myb-{rouge,orange,jaune,vert,bleu,violet}-ensps.pdf` in the same folder
(`myb_rouge.pdf` ... `myb_bleu.pdf`, `edu_11624.pdf` for violet). Teams: rouge YSGC
Building Workshop, orange Dupond et Du Pont, jaune Pontcepteur, vert Pont Pidur, bleu Pont
DicheryCo&Co, violet Popsicle Bridge (myb_main p. 3).

Sticks: one colour per team, dyed or painted (myb_main p. 3; edu_14198 p. 4 says the dye
changed their properties; violet and myb_main say sanding the paint off raised the
strength of glued joints by 20 % and almost 30 %). Glue: wood glue ("colle à bois"); the photo of the
materials (myb_main Fig. 2) shows Pattex Bois-Hout Express. **Not white school glue.**
Dimensions: jaune measured 50 sticks (its Table 1, PDF p. 5): 114 +-1.4 x 9.6 +-1 x 2.02
+-0.4 mm, 630 +-12 kg/m^3, "intervalle de confiance de 95%" (whether of the mean or of the
sticks is not said; the +-9.3 mm^2 it gives on a 19.39 mm^2 section reads as a spread).
Machines: Instron 3369 (rouge, violet), Instron 3000 (jaune), bleu's unnamed.

What is in the files:

* **Laps** (rouge, ch. 3 section 4, PDF pp. 19-20): two sticks glued over 1/4, 1/3 and 1/2
  of their length and pulled in the Instron: 1300 N, the wood broke obliquely; 1700 N,
  "la colle se casse et non le bois"; 2000 N, the wood yielding ("écrouissage du bois").
  Only the 1700 N lap is a glue-line failure: it is `glue_laps.csv`, the only lap in the
  six reports with both a load and a glue failure. The other two are lower bounds for the
  glue, at most, in `other/wood_failed_laps.csv`. Clamping, sanding, cure and rate not
  stated; the team calls its results unreliable because of its grips. Orange (annex, PDF
  p. 23) pulled laps of 5-30 mm: "une longueur de collage de 1,5 cm suffit pour atteindre
  la résistance mécanique du matériau", no loads given; violet says most of its laps
  failed in the glue, no loads given.
* **Buckling**: bleu Table 4 (PDF p. 8): three single sticks and three I-beams with their
  loads, end conditions and a "longueur de flambement"; rouge (PDF p. 18): the mean of
  several pin-ended sticks, 90 N, length not stated; myb_main (p. 5): 100 N, conditions
  not stated. `other/buckling.csv`.
* **Means** of tension, bending, compression and edge-glued joints by team:
  `other/team_summaries.yaml`. They disagree widely (tension E 10.8-15.3 GPa, bending E
  5.0-7 GPa, strength 50-100 MPa), each from its own handful of sticks.

### 1.4 Move Your Bridge 2.0 (edu_14198)

C. Desodt, H. Horsin Molinaro, X. Jourdain, M. Poncelet, "Move Your Bridge 2.0 : Concevoir
et construire une maquette de pont mobile", ENS Paris-Saclay / Eduscol 14198, 21 Nov 2022.
https://sti.eduscol.education.fr/sites/eduscol.education.fr.sti/files/ressources/pedagogiques/14198/14198-move-your-bridge-20-concevoir-et-construire-une-maquette-de-pont-mobile-ensps.pdf

Undyed sticks, Pattex Bois-Hout Express (photo, label D2). Its Table 1 (p. 4) is the
2018-19 dimension table reprinted, not said to be measured again. Tableau 2 (p. 7): each
group's 5 % tension load, strength and E; Fig. 6 (p. 6): twelve tension tests, group not
stated (stress, E and load each; mean 2088 N, against 1389 N for the 76 of 1.1): both in
`other/team_summaries.yaml`, not in the coupon CSVs (another campaign, another strength).
Fig. 7-8 (histograms of 100 tension results, lap strength by overlap and sanding) are
charts only: not transcribed.

## 2. Cedarville University (ASEE 2022)

L. Fredette, M. Kennedy, "Work in Progress: Reformulation of a Truss Competition Course
Project to Improve Educational Outcomes", 2022 ASEE Annual Conference & Exposition, paper
ID 37551.
https://peer.asee.org/work-in-progress-reformulation-of-a-truss-competition-course-project-to-improve-educational-outcomes.pdf

Means and ranges only (PDF p. 6): 10 sticks in tension, 12.3 ksi (84.8 MPa) ultimate,
range 13.6 ksi; E 1219 ksi (8.40 GPa), range 579 ksi; 24 lap-shear specimens, half sanded,
0.495 ksi (3.41 MPa) unsanded and 0.783 ksi (5.40 MPa) sanded, ranges 85 % of the mean;
the laps failed in the glue (p. 8). The glue, the lap size, the sticks, the machine, the
rate and how E was measured are not stated (the list is in `cedarville_2022/summary.yaml`).
With no lap size, a lap strength cannot be simulated; with no individual values, nothing
here is a coupon file.

## 3. Ambiguities and contradictions in the sources

* **The lap that failed in its glue** (rouge): the text pairs 1/4, 1/3 and 1/2 of the stick
  with 1300, 1700 and 2000 N; Fig. 3.7 labels its curves 2.5, 3.5 and 5 cm and its peaks
  read about 1920, 1330 and 1760 N. By the text, the glue-failure lap is 1/3 of 114 mm =
  38 mm (4.7 MPa average); by the figure, the 1760 N curve is 50 mm (3.5 MPa). The text is
  the source's statement of what was tested and what broke; `glue_laps.csv` takes it (38
  mm) and its notes give the other reading. `literature_coupons.py` reports both.
* **Bleu's "longueur de flambement"** (46, 47, 109 mm): undefined. Its Euler column is
  pi^2 E I / l^2 with the listed l itself, so the team used it as the effective length;
  for the clamped sticks it may be the free length between the grips (then the effective
  length is half and the Euler load four times). Both readings are simulated.
* **edu_14235**: the text gives four results for three unglued sticks on edge (228, 156,
  130, 241 N), Table 1 three; Fig. 10 and Fig. 26, curves of on-edge tests, are captioned
  "axe faible".
* **Units**: jaune gives its compressive strength as "30 GPa" then "Rc = 30 MPa"; violet an
  on-edge bending stress of "1,25Pa": kept as printed in `team_summaries.yaml`, not used.
* **Years**: edu_14198 calls the dyed-stick edition "2019-2020"; myb_main, its own
  description, says 2018-2019 and is dated 2 Dec 2019.
* **Dimensions**: none of the ENS coupon campaigns (1.1, 1.2) gives its sticks' section;
  the 9.6 x 2.02 mm in the coupon CSVs is jaune's 2018-19 mean of 50 dyed sticks.
