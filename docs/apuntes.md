# Monte Carlo
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

# Gymnasium
