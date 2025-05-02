import pandas as pd
from .utils import str_to_complex, print_matrix

def load_admittance_matrix(filepath):
    """Carrega a matriz de admitância"""
    try:
        matriz = pd.read_excel(filepath, index_col=0, header=0)
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                matriz.iloc[i, j] = str_to_complex(matriz.iloc[i, j])
        
        print(f"Dimensões da matriz: {matriz.shape}")
        print_matrix(matriz, "Matriz de Admitância")
        return matriz
    except Exception as e:
        raise Exception(f"Erro ao carregar matriz de admitância: {e}")

def load_bus_data(filepath):
    """Carrega dados das barras"""
    try:
        dados = pd.read_excel(filepath, index_col=0, header=0)
        print("\nDados das barras carregados com sucesso!")
        return dados
    except Exception as e:
        raise Exception(f"Erro ao carregar dados das barras: {e}")

def load_impedance_data(filepath):
    """Carrega dados de impedância"""
    try:
        dados = pd.read_excel(filepath, index_col=0, header=0)
        print("\nDados de impedância carregados com sucesso!")
        return dados
    except Exception as e:
        raise Exception(f"Erro ao carregar dados de impedância: {e}")