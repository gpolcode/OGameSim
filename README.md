## Value To Beat in 8000 Steps
Roi: 670'054'360.798

## input parameter
player:
current metal, current crystal, current deut

plasma:
level
metal upgrade cost, crystal upgrade cost, deut upgrade cost
production [metal, crystal, deut]
upgraded production [metal, crystal, deut]

astro:
level
metal upgrade cost, crystal upgrade cost, deut upgrade cost

20x
n-planet (all zero if unresearched)
metal:
level
metal upgrade cost, crystal upgrade cost, deut upgrade cost
production [metal]
upgraded production [metal]

crystal:
level
metal upgrade cost, crystal upgrade cost, deut upgrade cost
production [crystal]
upgraded production [crystal]

deut:
level
metal upgrade cost, crystal upgrade cost, deut upgrade cost
production [deut]
upgraded production [deut]

## actions
proceed to next day
astro upgrade
plasma upgrade

20x
n-planet metal upgrade
n-planet crystal upgrade
n-planet deut upgrade

## termination
random 8000-12000 days

## rewards
positives:
- points 1:1

negative:
- action from unavailable planet -5% of points
- action with to high of a cost -5% of points
