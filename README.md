## Value To Beat in 8000 Steps
Roi: 266'316'720.384

## input parameter
player:
current metal, current crystal, current deut

astro:
level
metal upgrade cost, crystal upgrade cost, deut upgrade cost

plasma:
level
metal upgrade cost, crystal upgrade cost, deut upgrade cost
production [metal, crystal, deut]
upgraded production [metal, crystal, deut]

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

## input parameter v2
player:
current mse
todays mse prod

astro:
mse upgrade cost

plasma:
mse upgrade cost
upgraded production mse

20x
n-planet (all zero if unresearched)
metal:
mse upgrade cost
upgraded production mse

crystal:
mse upgrade cost
upgraded production mse

deut:
mse upgrade cost
upgraded production mse

## actions
proceed to next day
astro upgrade
plasma upgrade

20x
n-planet metal upgrade
n-planet crystal upgrade
n-planet deut upgrade

## termination
random 8000days
