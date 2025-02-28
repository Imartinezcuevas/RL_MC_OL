"""
Module: agentes/tile_coder.py
Description: Implementación de Tile Coding para aproximación de funciones.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/28

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
import gymnasium as gym

class TileCoder:
    """
    Implementación de Tile Coding para aproximación de funciones.
    """

    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 num_tilings: int = 8,
                 num_tiles: int = 8,
                 scale_factor: float = 1.0):
        """
        Inicializa el codificador de tiles
        
        Args:
            observation_space: Espacio de observación (continuo)
            num_tilings: Número de tilings a usar
            num_tiles: Número de tiles por dimensión
            scale_factor: Factor de escala para los límites
        """
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        
        # Dimensionalidad del espacio de observación
        self.dims = observation_space.shape[0]
        
        # Límites del espacio de observación
        self.low = observation_space.low * scale_factor
        self.high = observation_space.high * scale_factor
        
        # Tamaño del feature vector resultante
        self.n_features = num_tilings * (num_tiles ** self.dims)
        
        # Offset para cada tiling (desplazamiento)
        self.offsets = [np.random.uniform(0, 1.0 / self.num_tiles, self.dims) 
                         for _ in range(self.num_tilings)]
        
    def encode(self, observation: np.ndarray):
        """
        Codifica una observación en un vector de características sparse
        
        Args:
            observation: Vector de observación
            
        Returns:
            Vector de características con codificación one-hot
        """
        # Inicializar vector de características sparse (solo los índices activos)
        active_features = []
        
        # Normalizar observación al rango [0, num_tiles] para cada dimensión
        norm_obs = self.num_tiles * (observation - self.low) / (self.high - self.low)
        
        # Para cada tiling
        tiles_per_tiling = self.num_tiles ** self.dims
        for t in range(self.num_tilings):
            # Aplicar offset a la observación normalizada
            offset_obs = norm_obs + self.offsets[t]
            
            # Calcular índices de tile para cada dimensión
            tile_indices = np.floor(offset_obs).astype(int) % self.num_tiles
            
            # Calcular índice único para este tile en este tiling
            # Usamos codificación row-major para mapear coordenadas a índice
            index = 0
            for i, idx in enumerate(tile_indices):
                index += idx * (self.num_tiles ** i)
            
            # Añadir el índice global de esta característica
            active_features.append(t * tiles_per_tiling + index)
        
        return active_features