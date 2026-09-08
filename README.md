# Otkrivanje tema u korpusu vesti pomoću NMF faktorizacije

Projekat iz Naučnog izračunavanja, master studije, Matematički fakultet u Beogradu.
Bogdan Tomić.

Ideja je da se iz korpusa BBC vesti, bez ijedne oznake, samo faktorizacijom matrice izvuku teme,
pa da se dobijena podela uporedi sa stvarnim kategorijama. TF-IDF matrica i NMF su napisani od
nule u NumPy-u, a rezultati provereni protiv `scikit-learn`-a.

Ceo rad je u svesci `naucno_istrazivanje.ipynb`.

## Podaci

[BBC Full Text Document Classification](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification),
2225 članaka BBC-ja iz 2004. i 2005. godine, u pet foldera: business (510), entertainment (386),
politics (417), sport (511) i tech (401). Skup je izbalansiran, nasumično pogađanje daje 20%.

Podaci se ne čuvaju u repozitorijumu. Druga ćelija sveske ih preuzima preko Kaggle API-ja u
`./bbc_dataset`, za šta je potreban token `kaggle.json` u `~/.kaggle/`
(vidi [uputstvo](https://www.kaggle.com/docs/api)). Ako folder već postoji, preuzimanje se
preskače, pa se skup može i ručno raspakovati na tu putanju.

Oznaka kategorije je samo naziv foldera i koristi se tek u evaluaciji, nikada u modelu.

## Pokretanje

```
pip install -r requirements.txt
jupyter notebook naucno_istrazivanje.ipynb
```

Sveska se izvršava od vrha do dna (Restart & Run All). Sve je čist NumPy i SciPy, bez GPU-a.

## Tok rada

1. Preuzimanje podataka i osnovna analiza korpusa, broj i dužina dokumenata po kategoriji.
2. Tokenizacija: mala slova, izbacivanje interpunkcije, brojeva, stop-reči i tokena kraćih od tri
   slova, pa spajanje susednih sadržajnih reči u bigrame (`prime minister`, `bank of england`).
3. TF-IDF matrica u CSR formatu, sa pragovima `MIN_DF = 5` i `MAX_DF = 0.5`. Rečnik se svede sa
   227.793 na 12.798 tokena, matrica 2225 x 12798 je popunjena 1,17% (4 MB retko naspram 228 MB
   gusto).
4. NMF multiplikativnim pravilima (`nmf_mu`), najbolje od šest pokretanja za k = 5.
5. Uparivanje tema sa kategorijama Hungarian algoritmom, evaluacija i matrica konfuzije.
6. Semantička pretraga nad kolonama matrice H iz posebnog modela sa k = 40.

## Rezultati

| mera | vrednost |
| --- | --- |
| tačnost (posle Hungarian uparivanja) | 93,3% |
| NMI | 0,815 |
| ARI | 0,846 |
| nasumično pogađanje | 20,0% |

Pet tema koje model izdvoji poklapaju se sa pet kategorija:

| tema | uparena kategorija | tokeni sa najvećom težinom |
| --- | --- | --- |
| 1 | tech | mobile, people, music, technology, phone |
| 2 | business | us, growth, economy, bank, oil |
| 3 | politics | labour, election, blair, brown, party |
| 4 | entertainment | film, best, awards, award, actor |
| 5 | sport | england, game, win, wales, ireland |

Provere ispravnosti:

- sopstvena TF-IDF naspram `TfidfVectorizer`-a nad istim rečnikom i istim tokenima: najveća
  apsolutna razlika 5,6e-16, dakle mašinska preciznost `float64`
- sopstvena NMF naspram `sklearn.decomposition.NMF`: ista relativna greška (0,970), tačnost 93,3%
  naspram 92,4%, isto svrstavanje dokumenta u temu u 97,7% slučajeva

Greške su skoro sve između kategorija tech i entertainment i između politics i business.
Najveće ograničenje metode je osetljivost na početnu tačku: jedno od šest pokretanja (seme 0)
završi u lošijem lokalnom minimumu i daje 68,4% tačnosti, iako mu se greška rekonstrukcije
razlikuje tek u trećoj decimali. U sekciji 6.1 sveske to je i provereno, pokretanjem sa isključenim
uslovom zaustavljanja: ni posle 2000 iteracija greška i tačnost tog pokretanja se ne menjaju, jer
multiplikativna pravila ne mogu da razdvoje teme koje su se već slile.

## Literatura

- D. D. Lee, H. S. Seung, *Learning the parts of objects by non-negative matrix factorization*,
  Nature 401 (1999), 788-791.
- D. D. Lee, H. S. Seung, *Algorithms for Non-negative Matrix Factorization*, NIPS 13 (2001).
  Odatle su multiplikativna pravila koja su ovde implementirana.
- W. Xu, X. Liu, Y. Gong, *Document Clustering Based On Non-negative Matrix Factorization*,
  SIGIR 2003.
- D. Greene, P. Cunningham, *Practical Solutions to the Problem of Diagonal Dominance in Kernel
  Document Clustering*, ICML 2006. Rad uz koji je objavljen originalni BBC korpus.
- Dokumentacija `scikit-learn`-a:
  [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
  [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).
