# OGameSim

Dieses Repo versucht die theoretische Punkteanzahl für das Browserspiel OGame zu analysieren.

## Disclaimer
- Die aktuelle Implementation versucht die Punkteanzanzahl nach einer bestimmten Spiel zeit zu maximieren. Dies kann zu überoptimierung bzw. Nachteile nach dem Zeitfenster führen.
- Der aktuelle Ansatz ist möglicherweise nicht ein realistischer Usecase. Simuliert werden immer die Tag 0-n anstatt n1-n2.
- Die neuen Features von der Expansion Lifeforms sind einfachheitshalber nicht implementiert.

## Todo
- Weitere exploration rewards für level
- https://docs.ray.io/en/latest/tune/examples/pbt_guide.html
- Rewrite das die Game Simulation vollständig auf der GPU paralellisiert läuft

## Value To Beat in 8000 Steps
Roi: 266'316'720.384

## Input parameter
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

## Termination
8000days
