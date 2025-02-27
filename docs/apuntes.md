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

# Técnicas de control de aproximaciones
Las técnicas de control con aproximaciones en aprendizaje por refuerzo se utilizan cuando los métodos tabulares se vuelven inviables debido a un espacio de estados grande o continuo. En estos casos, en lugar de almacenar valores en una tabla, se utilizan funcones de aproximacion para estimar valores de acción o políticas.

En el contexto de aprendizaje por refuerzo, las funciones de aproximación permiten representar la funcon de valor $Q(s,a)$ sin necesidad de almacenar una tabla. Para esto, se usan métodos basados en funciones lineales o redes neuronales.

Las dos técnicas que nos interesan son:
* SARSA Semi-Gradiente
* Deep Q-Learning

## Temporal difference
TD(Temporal difference, o Diferenciasa temporales) es un método de aprendizaje en el que las estimaciones de los valroes de estado o acción se actualizan basandose en otras estimaciones en lugar de esperar una recompensa final.

"Es una combinación de Monte Carlo (usa recompensas reales a futuro) y Programación dinámica (usa estimaciones previas)."

**Fórmula de TD**
El error de TD($\delta$) mide la diferencia entre la predicción actual y una estimación mejorada basada en el siguiente estado:
$$\delta = r + \gamma V(s') - V(s)$$
* $r$: recompensa inmediata obtenida.
* $\gamma$: factor de descuento.
* $V(s')$: estimación del valor del próximo estado.
* $V(s)$: estimación del valor del estado actual.


## SARSA Semi-Gradiente
SARSA Semi-Gradiente es una versión de SARSA donde en lugar de almacenar valores en una tabla, se usa una funcion de aproximación lineal:
$$Q(s,a; w) \approx \sum_{i} w_i \cdot \phi_i (s,a)$$
donde:
* $w_i$ son los pesos aprendidos del modelo.
* $\phi_i (s,a)$ son características extraidas del estado y la acción (pueden ser funciones manuales o basadas en técnicas como Tile Coding).$

El algoritmo sigue la misma estructura de SARSA pero actualiza los pesos usando descenso de gradiente en lugar de tablas:
$$w \leftarrow w + \alpha \delta ∇_w Q(s,a;w)$$
donde el error $TD(\delta)$ se calcula como:
$$\delta = r + \gamma Q(s',a';w) - Q(s,a;w)$$

Esto tiene unas ventajas:
* Permite manejar espacios continuos.
* Más eficiente en problemas de alta dimensión que los métodos tabulares.

## Deep Q-Learning
DQN es una extensión de Q-Learning donde se usa una red neuronal para aproximar $Q(s,a)$:
$$Q(s,a;\theta) \approx Red$$
Los pesos $\theta$ de la red se actualizan minimizando el error de TD:
$$L(\theta) = E[(r+\gamma max_a Q(s',a';\theta^-)-Q(s,a;\theta)^2)]$$

donde:
* $\theta_-$ son los parámetros de una red objetivo.
* Se usa replay bujjer para almacenar experiencias y reducir correlación entre muestras.
* Se actualiza con descenso de gradiente.

## Tile Coding
El Tile Coding es una técnica de representación de estados usada en aprendizaje por refuerzo con funciones lineales. Permite transformar un espacio de estados continuo en una representación discreta que facilita el aprendizaje.

**¿Cómo funciona?**
* Se divide el espacio de estados en múltiples rejillas superuestas(tiles):
    * Se  crean varias rejillas desplazadas entre sí.
    * Cada rejilla cubre todo el espacio de estados, pero con un pequeño desplazamiento respecto a las otras.
* Activar tiles según el estado:
    * Cada estado cae en un tile dentro de cada rejilla.
    * Las representación de un estado será un vector binario donde las posiciones activadas son 1, y las demás son 0.
* Aproximación lineal con Tile Coding:
    * En lugar de tener una gran tabla Q(s,a) tenemos un vector de pesos w.
    * Para estimar Q(s,a), simplemente sumamos los pesos de lso tiles activados.
    * Durante el aprendizaje, los pesos se ajustan con descenso de gradiente.

Esto permite aproximar funciones de valores en espacios continuos sin usar redes neuronales.

**Ejemplo MountainCar**
El entorno MountainCar de Gymnasium tiene un espacio de estados continuo:
* s = (posición, velocidad)
* Acciones: $a \in$ {empujar izquierda, no empujar, empujar derecha}