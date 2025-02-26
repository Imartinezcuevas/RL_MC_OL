# Métodos tabulares

## Monte Carlo
Monte Carlo es un método basado en simulaciones para resolver problemas de Aprendizaje por refuerzo. A diferencia de otros métodos como diferencias temprales, Monte Carlo requiere episodios completos para actualizar los valores de estado o de acción.

Como funciona:
1. Explora el entorno: el agente interactúa con el entorno siguiendo una política y genera episodios completos hasta alcanzar un estado terminal.
2. Recompensa acumulada: para cada estado visitado, se calcula la recompensa acumulada desde ese punto hasta el final del episodio.
3. Actualizar valores: se actualizan los valores estimados de los estados basándose en la media de las recompensas obtenidas en episodios anteriores.

Hay dos variantes principales:
* First-visit: solo actualiza el valor de un estado la primera vez que aparece en un episodio.
* Every-visit: actualiza el valor de un estado cada vez que aparece en un episodio.

Como se actualiza la politica:
* On-Policy: aprende y actualiza su política mientras la usa para actuar en el entorno. Es decir, la misma política que genera los episodios se mejora con el tiempo.
* Off-Policy: aprende una política óptima diferente de la que usa para explorar el entorno. Se usa una política de comportamiento para recolertar datos y una política objetivo para aprender.

## SARSA
SARDA es un método de aprendizaje temporal-diferenciado (TD-learning) que actualiza los valores de una funcion de acción-valor $Q(s,a)$, que representa la utilidad esperada de  tomar una acción $a$ en un estado $s$. Funciona con la siguiente equación de actualización:
$$Q(s,a) <- Q(s,a) + \alpha [r + \gamma Q(s', a') - Q(s,a)]$$
Donde:
* $s$ y $a$ son el estado y la acción actuales.
* $r$ es la recompensa obtenida tras tomar la acción $a$.
* $s'$ es el nuevo estado tras la acción.
* $a'$ es la  nueva acción elegida en $s'$ siguiendo la política.
* $\alpha$ es la tasa de aprendizaje.
* $\gamma$ es el factor de descuento.

Pasos
1. Inicializar la funcion de valores $Q(s,a)$ arbitrariamente.
2. Elegir una acción $a$ en el estado $s$ usando una politica $\pi$.
3. Ejecutar la acción $a$ y observar la recompensa $r$ y el nuevo estado $s'$.
4. Elegir la proxima acción $a'$ en $s'$ según la política.
5. Actualizar el valor $Q(s,a)$ con la ecuacion de actualización de SARSA.
6. Repetir hasta alcanzar una condición de parada.

## Q-Learning
Q-Learning es un algoritmo de aprendizaje por refuerzo off-policy basado en valores. A diferencia de SARSA, Q-Learning aprende a partir de la mejor acción posible, sin importar cúal haya sido realmente seleccionada.

$$Q(s,a) <- Q(s,a) + \alpha [r + \gamma max_{a'} Q(s', a') - Q(s,a)] $$
Donde:
* $s$ y $a$ son el estado y la acción actuales.
* $r$ es la recompensa obtenida tras tomar la acción $a$.
* $s'$ es el nuevo estado tras la acción.
* $a'$ es la  nueva acción elegida en $s'$ siguiendo la política.
* $\alpha$ es la tasa de aprendizaje.
* $\gamma$ es el factor de descuento.
* $ max_{a'} Q(s', a')$ representa la mejor acción posible en el proximo estado s', lo que hace que Q-Learning sea off-policy (ya que no sigue necesariamente la política usada para actuar).