import time
import numpy as np  # Adicione esta linha no topo do arquivo
from lib.file_reader import load_admittance_matrix, load_bus_data, load_impedance_data
from lib.gauss_seidel import solve_power_flow
from lib.power_calculations import calculate_power_flows
from lib.utils import format_complex

def main():
    start_time = time.time()
    
    # Carregar dados
    try:
        matriz_admt = load_admittance_matrix("metodo_gauss_seidel/data/Matriz Admitância.xlsx")
        tipo_barras = load_bus_data("metodo_gauss_seidel/data/Barras.xlsx")
        impedancias = load_impedance_data("metodo_gauss_seidel/data/impedância.xlsx")
    except Exception as e:
        print(f"\nErro: {e}")
        return

    # Verificar consistência
    n_barras = len(tipo_barras)
    if matriz_admt.shape[0] != n_barras or matriz_admt.shape[1] != n_barras:
        print(f"\nErro: A matriz deve ser {n_barras}x{n_barras}")
        return

    # Resolver fluxo de carga
    print("\nIniciando cálculo do fluxo de carga...")
    vetor_tensao, iteracoes, erro = solve_power_flow(matriz_admt, tipo_barras, impedancias)
    
    # Resultados
    print(f"\nTempo de execução: {time.time() - start_time:.2f} segundos")
    print(f"\nConvergiu após {iteracoes} iterações com erro: {erro:.8f}")

    # Tensões
    print("\nTensões nas barras:")
    for i, tensao in enumerate(vetor_tensao):
        print(f"Barra {i+1}: {format_complex(tensao)} pu | {abs(tensao):.3f} pu ∠ {np.degrees(np.angle(tensao)):.3f}°")

    # Cálculos de potência
    resultados = calculate_power_flows(vetor_tensao, matriz_admt, tipo_barras, impedancias)

    # Potências geradas
    print("\nPotências geradas:")
    for i in range(n_barras):
        print(f"Barra {i+1}: P = {resultados['P_gerada'][i]*100:.2f} MW | Q = {resultados['Q_gerada'][i]*100:.2f} MVar")

    # Fluxos nas linhas
    print("\nFluxos nas linhas:")
    for i in range(len(impedancias)):
        de = int(impedancias.iloc[i]["DE"])
        para = int(impedancias.iloc[i]["PARA"])
        print(f"Linha {de}-{para}: P = {resultados['fluxos_ativos'][i]*100:.2f} MW | Q = {resultados['fluxos_reativos'][i]*100:.2f} MVar")

    # Perdas
    print("\nPerdas totais:")
    print(f"P = {sum(resultados['perdas_ativas'])*100:.4f} MW")
    print(f"Q = {sum(resultados['perdas_reativas'])*100:.4f} MVar")

if __name__ == "__main__":
    main()