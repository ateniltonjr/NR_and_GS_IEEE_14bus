import cmath as cmt
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class NR:
    def __init__(self):
        self.__Sbase = 100e6       # 100 MVA Base
        self.__dados = dict()      # Dicionário para armazenar os dados das barras
        self.__Sesp = dict()       # Potências especificadas
        self.__Sbarras = dict()    # Potências calculadas nas barras
        self.__ligacoes = dict()   # Ligações entre barras
        self.__ybus = None         # Matriz admitância
        self.__V = dict()          # Tensões complexas
        self.__I = dict()          # Correntes
        self.__fluxS = dict()      # Fluxos de potência
        self.__Perdas = 0          # Perdas do sistema

        # Para plotagem
        self.__tensaoPlot = dict()
        self.__angPlot = dict()
        
        self.nPV = 0               # Número de barras PV
        self.nPQ = 0               # Número de barras PQ
        self.count = 0             # Contador de iterações

        # Submatrizes jacobianas
        self.__J1 = None
        self.__J2 = None
        self.__J3 = None
        self.__J4 = None
        self.__Jacobiana = None

    def setBarras(self, barra, code, tensao, ang, carga, geracao):
        self.__dados[barra] = {
            'code': code, 
            'tensao': tensao,
            'ang': mt.radians(ang), 
            'carga': complex(carga)/self.__Sbase,
            'geracao': complex(geracao)/self.__Sbase
        }
        self.__tensaoPlot[barra] = [tensao]
        self.__angPlot[barra] = [ang]
    
    def printBarras(self):
        print('\n===================================DADOS DAS BARRAS===================================')
        print('Sbase = ', self.__Sbase, 'VA')
        for i in sorted(self.__dados.keys()):
            print(f"Barra {i}: {self.__dados[i]}")
        print('=======================================================================================')

    def setSesp(self):
        for i in self.__dados:
            if self.__dados[i]['code'] == 2:   # Barra PQ
                self.__Sesp[i] = {
                    'Pesp': (self.__dados[i]['geracao'] - self.__dados[i]['carga']).real,
                    'Qesp': (self.__dados[i]['geracao'] - self.__dados[i]['carga']).imag
                }
            elif self.__dados[i]['code'] == 3: # Barra PV
                self.__Sesp[i] = {
                    'Pesp': (self.__dados[i]['geracao'] - self.__dados[i]['carga']).real,
                    'Qesp': None
                }
            elif self.__dados[i]['code'] == 1: # Barra Slack
                self.__Sesp[i] = {
                    'Pesp': None,
                    'Qesp': None
                }
        
        print('\n============================POTÊNCIAS ESPECIFICADAS============================')
        for i in sorted(self.__Sesp.keys()):
            print(f"Barra {i}: {self.__Sesp[i]}")
        print('================================================================================')

    def setLigacoes(self, b1, b2, impedancia=None, admitancia=None):
        if impedancia is not None:
            admitancia = 1 / complex(impedancia)
        elif admitancia is not None:
            impedancia = 1 / complex(admitancia)

        self.__ligacoes[(b1, b2)] = {
            'impedancia': complex(impedancia),
            'admitancia': complex(admitancia)
        }
        
    def printLigacoes(self):
        print('\n===================================LIGAÇÕES===================================')
        for i in sorted(self.__ligacoes.keys()):
            print(f'Ligação {i}: {self.__ligacoes[i]}')
        print('=============================================================================')

    def __printYbus(self):
        print('\n================================MATRIZ ADMITÂNCIA================================')
        for i, row in enumerate(self.__ybus):
            print(f"Linha {i+1}:", " ".join(f"{elem:10.6f}" for elem in row))
        print('================================================================================')

    def Ybus(self):
        n_barras = len(self.__dados)
        self.__ybus = np.zeros((n_barras, n_barras), dtype=complex)

        # Preenche os elementos fora da diagonal
        for (b1, b2), dados in self.__ligacoes.items():
            y = dados['admitancia']
            self.__ybus[b1-1][b2-1] -= y
            self.__ybus[b2-1][b1-1] -= y

        # Preenche os elementos da diagonal
        for i in range(n_barras):
            self.__ybus[i][i] = -sum(self.__ybus[i])

        # Conta barras PQ e PV
        self.nPQ = sum(1 for barra in self.__dados.values() if barra['code'] == 2)
        self.nPV = sum(1 for barra in self.__dados.values() if barra['code'] == 3)

        self.__printYbus()

    def Sinjetada(self):
        self.__deltaPeQ = []
        self.__ResiduoP = []
        self.__ResiduoQ = []

        barras_ord = sorted(self.__dados.keys())
        
        for i in barras_ord:
            if self.__dados[i]['code'] == 1:  # Barra slack
                continue
                
            P_calc = 0
            Q_calc = 0
            V_i = self.__dados[i]['tensao']
            theta_i = self.__dados[i]['ang']
            
            for j in barras_ord:
                V_j = self.__dados[j]['tensao']
                theta_j = self.__dados[j]['ang']
                Y_ij = self.__ybus[i-1][j-1]
                G_ij = Y_ij.real
                B_ij = Y_ij.imag
                theta_ij = theta_i - theta_j
                
                P_calc += V_i * V_j * (G_ij * mt.cos(theta_ij) + B_ij * mt.sin(theta_ij))
                Q_calc += V_i * V_j * (G_ij * mt.sin(theta_ij) - B_ij * mt.cos(theta_ij))
            
            # Calcula resíduos
            if self.__dados[i]['code'] == 2:  # Barra PQ
                P_esp = self.__Sesp[i]['Pesp']
                Q_esp = self.__Sesp[i]['Qesp']
                self.__ResiduoP.append(P_esp - P_calc)
                self.__ResiduoQ.append(Q_esp - Q_calc)
            elif self.__dados[i]['code'] == 3:  # Barra PV
                P_esp = self.__Sesp[i]['Pesp']
                self.__ResiduoP.append(P_esp - P_calc)
                # Para barras PV, guardamos Q calculado para usar depois
                self.__Sesp[i]['Qcalc'] = Q_calc

        # Concatena resíduos P e Q
        self.__deltaPeQ = self.__ResiduoP + self.__ResiduoQ
        
        print('\n===================================RESÍDUOS===================================')
        print("Resíduos P:", self.__ResiduoP)
        print("Resíduos Q:", self.__ResiduoQ)
        print('=============================================================================')

    def __calcularJ1(self, list_ang):
        n = len(list_ang)
        self.__J1 = np.zeros((n, n))
        
        barras_ord = sorted(self.__dados.keys())
        
        for k, i in enumerate(list_ang):
            for m, j in enumerate(list_ang):
                V_i = self.__dados[i]['tensao']
                theta_i = self.__dados[i]['ang']
                
                if i == j:  # Elemento diagonal
                    sum_term = 0
                    for l in barras_ord:
                        if l == i:
                            continue
                        V_l = self.__dados[l]['tensao']
                        theta_l = self.__dados[l]['ang']
                        Y_il = self.__ybus[i-1][l-1]
                        theta_il = theta_i - theta_l
                        
                        sum_term += V_i * V_l * (Y_il.real * mt.sin(theta_il) - Y_il.imag * mt.cos(theta_il))
                    
                    self.__J1[k][m] = -sum_term
                else:  # Elemento fora da diagonal
                    V_j = self.__dados[j]['tensao']
                    theta_j = self.__dados[j]['ang']
                    Y_ij = self.__ybus[i-1][j-1]
                    theta_ij = theta_i - theta_j
                    
                    self.__J1[k][m] = V_i * V_j * (Y_ij.real * mt.sin(theta_ij) - Y_ij.imag * mt.cos(theta_ij))
        
        return self.__J1
    
    def __calcularJ2(self, list_tensao, list_ang):
        n_ang = len(list_ang)
        n_tensao = len(list_tensao)
        self.__J2 = np.zeros((n_ang, n_tensao))
        
        barras_ord = sorted(self.__dados.keys())
        
        for k, i in enumerate(list_ang):
            for m, j in enumerate(list_tensao):
                V_i = self.__dados[i]['tensao']
                theta_i = self.__dados[i]['ang']
                
                if i == j:  # Elemento diagonal
                    sum_term = 0
                    for l in barras_ord:
                        if l == i:
                            continue
                        V_l = self.__dados[l]['tensao']
                        theta_l = self.__dados[l]['ang']
                        Y_il = self.__ybus[i-1][l-1]
                        theta_il = theta_i - theta_l
                        
                        sum_term += V_l * (Y_il.real * mt.cos(theta_il) + Y_il.imag * mt.sin(theta_il))
                    
                    self.__J2[k][m] = V_i * (2 * self.__ybus[i-1][i-1].real + sum_term)
                else:  # Elemento fora da diagonal
                    V_j = self.__dados[j]['tensao']
                    theta_j = self.__dados[j]['ang']
                    Y_ij = self.__ybus[i-1][j-1]
                    theta_ij = theta_i - theta_j
                    
                    self.__J2[k][m] = V_i * (Y_ij.real * mt.cos(theta_ij) + Y_ij.imag * mt.sin(theta_ij))
        
        return self.__J2
    
    def __calcularJ3(self, list_tensao, list_ang):
        n_tensao = len(list_tensao)
        n_ang = len(list_ang)
        self.__J3 = np.zeros((n_tensao, n_ang))
        
        barras_ord = sorted(self.__dados.keys())
        
        for k, i in enumerate(list_tensao):
            for m, j in enumerate(list_ang):
                V_i = self.__dados[i]['tensao']
                theta_i = self.__dados[i]['ang']
                
                if i == j:  # Elemento diagonal
                    sum_term = 0
                    for l in barras_ord:
                        if l == i:
                            continue
                        V_l = self.__dados[l]['tensao']
                        theta_l = self.__dados[l]['ang']
                        Y_il = self.__ybus[i-1][l-1]
                        theta_il = theta_i - theta_l
                        
                        sum_term += V_l * (Y_il.real * mt.sin(theta_il) - Y_il.imag * mt.cos(theta_il))
                    
                    self.__J3[k][m] = -sum_term
                else:  # Elemento fora da diagonal
                    V_j = self.__dados[j]['tensao']
                    theta_j = self.__dados[j]['ang']
                    Y_ij = self.__ybus[i-1][j-1]
                    theta_ij = theta_i - theta_j
                    
                    self.__J3[k][m] = -V_i * (Y_ij.real * mt.sin(theta_ij) - Y_ij.imag * mt.cos(theta_ij))
        
        return self.__J3
    
    def __calcularJ4(self, list_tensao, list_ang):
        n = len(list_tensao)
        self.__J4 = np.zeros((n, n))
        
        barras_ord = sorted(self.__dados.keys())
        
        for k, i in enumerate(list_tensao):
            for m, j in enumerate(list_tensao):
                V_i = self.__dados[i]['tensao']
                theta_i = self.__dados[i]['ang']
                
                if i == j:  # Elemento diagonal
                    sum_term = 0
                    for l in barras_ord:
                        if l == i:
                            continue
                        V_l = self.__dados[l]['tensao']
                        theta_l = self.__dados[l]['ang']
                        Y_il = self.__ybus[i-1][l-1]
                        theta_il = theta_i - theta_l
                        
                        sum_term += V_l * (Y_il.real * mt.sin(theta_il) - Y_il.imag * mt.cos(theta_il))
                    
                    self.__J4[k][m] = -2 * V_i * self.__ybus[i-1][i-1].imag - sum_term
                else:  # Elemento fora da diagonal
                    V_j = self.__dados[j]['tensao']
                    theta_j = self.__dados[j]['ang']
                    Y_ij = self.__ybus[i-1][j-1]
                    theta_ij = theta_i - theta_j
                    
                    self.__J4[k][m] = -V_i * (Y_ij.real * mt.sin(theta_ij) - Y_ij.imag * mt.cos(theta_ij))
        
        return self.__J4
    
    def setJacobiana(self):
        # Identifica automaticamente as barras PQ e PV
        list_tensao = [b for b in self.__dados if self.__dados[b]['code'] == 2]  # Barras PQ
        list_ang = [b for b in self.__dados if self.__dados[b]['code'] in [2, 3]]  # Barras PQ e PV

        J1 = self.__calcularJ1(list_ang)
        J2 = self.__calcularJ2(list_tensao, list_ang)
        J3 = self.__calcularJ3(list_tensao, list_ang)
        J4 = self.__calcularJ4(list_tensao, list_ang)

        # Monta a matriz Jacobiana completa
        self.__Jacobiana = np.block([
            [J1, J2],
            [J3, J4]
        ])
        
        print('\n===================================JACOBIANA===================================')
        for row in self.__Jacobiana:
            print(" ".join(f"{elem:10.6f}" for elem in row))
        print('================================================================================')
        
        return self.__Jacobiana

    def linearSolver(self):
        try:
            # Resolve o sistema linear J * Δx = -ΔS
            delta_x = np.linalg.solve(self.__Jacobiana, -np.array(self.__deltaPeQ))
            
            # Separa as correções de ângulo e tensão
            n_ang = len([b for b in self.__dados if self.__dados[b]['code'] in [2, 3]])
            delta_theta = delta_x[:n_ang]
            delta_V = delta_x[n_ang:]
            
            # Aplica as correções
            idx_ang = 0
            idx_V = 0
            for barra in sorted(self.__dados.keys()):
                if self.__dados[barra]['code'] in [2, 3]:  # Barras PQ e PV
                    self.__dados[barra]['ang'] += delta_theta[idx_ang]
                    self.__angPlot[barra].append(mt.degrees(self.__dados[barra]['ang']))
                    idx_ang += 1
                
                if self.__dados[barra]['code'] == 2:  # Barras PQ
                    self.__dados[barra]['tensao'] += delta_V[idx_V]
                    self.__tensaoPlot[barra].append(self.__dados[barra]['tensao'])
                    idx_V += 1
                
            return True
        except np.linalg.LinAlgError as e:
            print(f"Erro na solução do sistema linear: {str(e)}")
            return False

    def novaInjecao(self):
        # Atualiza a injeção de potência na barra slack
        slack_bus = next((b for b in self.__dados if self.__dados[b]['code'] == 1), None)
        if slack_bus:
            P_calc = 0
            Q_calc = 0
            V_i = self.__dados[slack_bus]['tensao']
            theta_i = self.__dados[slack_bus]['ang']
            
            for j in self.__dados:
                V_j = self.__dados[j]['tensao']
                theta_j = self.__dados[j]['ang']
                Y_ij = self.__ybus[slack_bus-1][j-1]
                theta_ij = theta_i - theta_j
                
                P_calc += V_i * V_j * (Y_ij.real * mt.cos(theta_ij) + Y_ij.imag * mt.sin(theta_ij))
                Q_calc += V_i * V_j * (Y_ij.real * mt.sin(theta_ij) - Y_ij.imag * mt.cos(theta_ij))
            
            self.__Sbarras[slack_bus] = {'P': P_calc, 'Q': Q_calc}
            self.__dados[slack_bus]['geracao'] = P_calc + 1j * Q_calc

        # Atualiza a potência reativa nas barras PV
        for barra in self.__dados:
            if self.__dados[barra]['code'] == 3:  # Barra PV
                Q_calc = self.__Sesp[barra]['Qcalc']
                self.__Sbarras[barra] = {'Q': Q_calc}
                # Mantém a geração de P especificada e atualiza Q
                P_ger = np.real(self.__dados[barra]['geracao'])
                self.__dados[barra]['geracao'] = P_ger + 1j * Q_calc
    
    def solveCircuito(self, erro=1e-5, iteracoes_max=50):
        self.count = 0
        
        # Passos iniciais
        self.Ybus()
        self.setSesp()
        
        convergiu = False
        while not convergiu and self.count < iteracoes_max:
            self.count += 1
            
            # Calcula resíduos
            self.Sinjetada()
            
            # Verifica convergência
            max_residuo = max(abs(np.array(self.__deltaPeQ)))
            if max_residuo < erro:
                convergiu = True
                break
                
            # Monta e resolve sistema linear
            self.setJacobiana()
            if not self.linearSolver():
                print("Erro na solução do sistema linear. Parando iterações.")
                break
                
            # Atualiza injeções
            self.novaInjecao()
        
        if convergiu:
            print(f'\nConvergiu em {self.count} iterações com erro máximo de {max_residuo:.6f}')
        else:
            print(f'\nNão convergiu após {self.count} iterações. Erro máximo: {max_residuo:.6f}')

    def Tensoes(self, print_flag=False):
        self.__V = {}
        for barra in sorted(self.__dados.keys()):
            V = self.__dados[barra]['tensao']
            theta = self.__dados[barra]['ang']
            self.__V[barra] = cmt.rect(V, theta)
            
            if print_flag:
                print(f"Barra {barra}: {V:.6f} ∠ {mt.degrees(theta):.2f}°")
        return self.__V

    def Correntes(self, print_flag=False):
        self.__I = {}
        V = self.Tensoes()
        
        # Calcula correntes nas linhas
        for (b1, b2) in self.__ligacoes:
            Y = self.__ligacoes[(b1, b2)]['admitancia']
            self.__I[(b1, b2)] = (V[b1] - V[b2]) * Y
            self.__I[(b2, b1)] = -self.__I[(b1, b2)]
            
            if print_flag:
                I_mag = abs(self.__I[(b1, b2)])
                I_ang = mt.degrees(cmt.phase(self.__I[(b1, b2)]))
                print(f"Corrente {b1}-{b2}: {I_mag:.6f} ∠ {I_ang:.2f}°")
        
        return self.__I

    def fluxoS(self, print_flag=False):
        self.__fluxS = {}
        V = self.Tensoes(print_flag)
        I = self.Correntes(print_flag)
        
        # Calcula fluxos de potência
        for (b1, b2) in self.__ligacoes:
            S = V[b1] * np.conj(I[(b1, b2)])
            self.__fluxS[(b1, b2)] = S
            
            if print_flag:
                print(f"Fluxo {b1}-{b2}: P = {S.real:.6f} pu, Q = {S.imag:.6f} pu")
        
        return self.__fluxS

    def Perdas(self):
        perdas_totais = 0
        processed = set()
        
        for (b1, b2) in self.__fluxS:
            if (b1, b2) in processed or (b2, b1) in processed:
                continue
                
            S_ij = self.__fluxS.get((b1, b2), 0)
            S_ji = self.__fluxS.get((b2, b1), 0)
            perda = (S_ij + S_ji).real
            
            if abs(perda) < 1e6:  # Filtra valores absurdos
                perdas_totais += perda
                processed.add((b1, b2))
        
        print(f"\nPerdas Totais: {perdas_totais:.6f} pu")
        return perdas_totais

    def plotDados(self, tensao=False, ang=False):
        if tensao or ang:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            if tensao:
                for barra in sorted(self.__tensaoPlot.keys()):
                    if self.__dados[barra]['code'] == 2:  # Barras PQ
                        ax1.plot(self.__tensaoPlot[barra], 
                                marker='o', linestyle='-', 
                                label=f'Barra {barra}')
                
                ax1.set_title('Variação de Tensão nas Barras PQ', pad=20)
                ax1.set_xlabel('Número de Iterações')
                ax1.set_ylabel('Tensão [pu]')
                ax1.set_ylim(0.95, 1.05)
                ax1.grid(True, linestyle='--', alpha=0.5)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            if ang:
                for barra in sorted(self.__angPlot.keys()):
                    if self.__dados[barra]['code'] != 1:  # Barras PQ e PV
                        ax2.plot(self.__angPlot[barra], 
                                marker='s', linestyle='--', 
                                label=f'Barra {barra}')
                
                ax2.set_title('Variação do Ângulo nas Barras', pad=20)
                ax2.set_xlabel('Número de Iterações')
                ax2.set_ylabel('Ângulo [graus]')
                ax2.grid(True, linestyle='--', alpha=0.5)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.show()