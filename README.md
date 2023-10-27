# syn-word-para-swap
This repo contains 3 algorithms developed by me to automatically obfuscate articles in different ways. Currently the detectors do not work as they utilized Open AI's detector which was taken down in July 2023.

# Synonym Swap Result:
## Human Generated Text (Previous Label: Very Unlikely AI gener-
ated, New Label: Unclear if it is)
Human Text (Original): “a look at some of donald trump’s early activ-
ity as president: – 24: executive orders and memoranda signed. that includes
to withdraw the nited states from trade deal, impose federal hiring freeze re-
duce regulations related health care law enacted under former president barack
obama. 1: blocked. an order ban travelers. . . ”
## Synonym Swap Text: ”a look at some of donald trump’s early activity as
President of the United States – 24: executive orders and memoranda signed.
that includes to withdraw the nited states from trade deal, impose federal hir-
ing stop dead reduce regulations related health care law enacted under former
president barack obama. 1: hinder an order ban travelers. . . 

# UID Word Swap Results:
## Human Generated Text:
Original Article (Human): “washington ( ) at a time when president donald
trump seems to permeate nearly every aspect of american discourse, it might
come as surprise that the first movie from barack and michelle obama’s pro-
duction company, higher ground, never mentions him by name.but subtlety is
part power factory, new netflix documentary charts r ening factory in dayton,
ohio. over course two hours, movie, directed seasoned documentarians steven
bognar julia reichert, serves quiet historical political corrective, offering their
portrait state america’s industrial heartland prodding viewers rethink who, ex-
actly, project.american starts on december 23, 2008, crowd gathers learn gen-
eral motors plant dayton has shuttered. then fast-forwards 2015, ens enterprise,
fuyao glass america, arm shanghai-based company manufactures automotive
glass. one man makes fuyao’s expanded mission crystal clear: what we’re do-
ing melding cultures together: chinese culture s culture. so we are truly global
organization.as some critics have pointed out, is, important ways, commen-
tary unpredictability globalization; york times review frames underscoring haves
have-nots.but it’s also much more than that. arrives moment white house con-
tinues make vociferous, bold claims about economy, particularly manufacturing.
that’s despite increasing concerns economists warnings history recession could
be horizon. there’s sobering contrast between trump’s rhetoric how job growth
ballooned during his presidency reality broader slowdown slamming states –
including ohio helped win 2016 presidential election.read”

## UID Word Swap Text UID Score variance & difference2: 
“ washington ( ) at a time when president donald trump seems to permeate nearly every
aspect of american discourse , it might come as surprise that the first movie
from barack and michelle obama ’s publishing company , higher ground , never
mentions him by name.but subtlety is part power factory , new netflix doc-
umentary charts r ening factory in dayton , ohio . over course two hours ,
movie , directed seasoned documentarians steven bognar julia reichert , serves
quiet historical political corrective , offering their portrait state america ’s in-
dustrial heartland helps viewers rethink who , exactly , project.american starts
on december 23 , 2008 , crowd gathers learn general motors plant dayton has
shuttered . then fast-forwards 2015 , ens enterprise , fuyao glass corporation ,
arm shanghai-based company manufactures automotive glass . one man makes
fuyao ’s expanded mission crystal clear : what we ’re doing bringing cultures
together : chinese culture s culture . so we are truly global organization.as some
critics have pointed out , is , important ways , commentary upon globalization ;
york times review frames underscoring haves have-nots.but it ’s also much more
than that . arrives moment white house continues make vociferous , conflicting
claims about economy , particularly manufacturing . that ’s despite increas-
ing concerns economists thought history recession could be horizon . there ’s
sobering contrast between trump ’s rhetoric how job growth ballooned during
his political reality broader slowdown slamming states – including ohio helped
win 2016 presidential election.read”

# Paraphrase UID Swap:
## Human Text (Original): 
“the nited nations has ended a campaign featuring wonder woman as an ambassador for women and girls, two months after
announcement was met with protests petition complaining that fictional su-
perhero inappropriate choice to represent female empowerment. in announcing
october, said it about girls everywhere, who are their own right, men boys sup-
port struggle gender equality. but not everyone saw way. nearly 45, 000 p
le signed protesting selection. a white of impossible proportions, scantily clad
shimmery, body suit american flag motif boots is appropriate spokeswoman
equity at nations, said. jeffrey brez, spokesman .”

## UID Paraphrase UID Score variance: 
“A campaign involving wonder woman as an ambassador for women and girls has ended two months after
being met with protests petition claiming that the fictional superhero inappro-
priate choice to represent female liberation. Girls everywhere, who are their
own right, and men boys support gender discrimination are all in announcing
october.. but not everyone saw way. More than half of the population of nearly
45, 000 p les have been rejected in the protesting vote. According to a white of
insignificant proportions, scantily clad shimmery, body suit american flag motif
boots, is appropriate spokeswoman equity in nations.. jeffrey brez, spokesman
.”

## UID Paraphrase UID Score Difference2: 
“Two months after being met with protests petition claiming that the fictional superhero inappropriate choice
to represent female liberation, the nited nations has ended a campaign starring
wonder woman as an ambassador for women and girls. Girls everywhere, who
are their own right, and men boys support gender discrimination are all in an-
nouncing october. but not everyone saw way. Over 55,000 people were rejected
in the protesting vote, with nearly 45, 000 p le signed protesting selection. Ac-
cording to the author, a white of insignificant proportions, scantily clad, body
suit american flag motif boots, is appropriate spokeswoman equity in nations.
jeffrey brez, spokesman ."

# Here is an example of the UID Paraphrase generating nonsensical results:

## Original Article: ’the plan for mandatory debt-market defaulting in the life
of dealer during first year rbi may postpone its to make compulsory all defaults
debt market when it is launched, chief economic adviser arvind subramanian
today said he told reporters that not be feasible have default one yearmarket
expectations rs 50 rate cut soon; raghuram rajan’s wish list yet revealed: shah -
business standardreserve bank governor rajan india and philip clarke, ceo tesco k
expectation a basis points next monetary policy meets with consensus, according
standard report as growth known, participants eagerly wait decisionrbi set 1 2
2% over five years: shaktikanta das linerbi line ’s key variable repo rate, which
displayed an increasing trend period 2011-14 declined 2015 following easing :
reserve moneycontrol.comrbi displayed’

## UID Paraphrase Variance Selected Article: According to a shakti bank
governor’s mandatory debt-market defaulting in the first year of the bank’s life,
the plan for mandatory debt-market defaulting in the life of the dealer during
the first year rbi’s first year may postponed, but the main variable repo rate,
which increased trend rate in 2015, remains unchanged, according to reserve
moneycontrol.comrbi’s forecast, rising trend rate remains unchanged, with in-
creasing trend rate declines rajan rbi a rajan rajan rajan rajan rajan rbi rbi
rajan rajan rajan rajan rbi rbi rbi’s first year rbi rbi rbi rbi rbi rbi rbi rbi rbi
rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi
rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi
rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi rbi
rbi rbi rbi rbi rbi rbi
